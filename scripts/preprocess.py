"""Priprema slike za klasifikator: ista pravila za keš za trening, proveru i kasnije API.

Redosled: EXIF rotacija, RGB, izbor okvira psa, kvadrat oko psa (ili rezervni kvadrat kad psa nema),
isecanje iz originala, dopuna srednjom bojom kad kvadrat izlazi iz slike, smanjenje na 224 px.

Modul ne zavisi od PyTorch-a. Detekcije dobija spolja (iz detektora ili iz data/detections.csv).
"""

from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, ImageOps

IMAGE_SIZE = 224               # ulaz klasifikatora: IMAGE_SIZE x IMAGE_SIZE
DOG_SCORE_THRESHOLD = 0.5      # okvir ispod ove pouzdanosti ne koristimo
BOX_MARGIN = 0.10              # kvadrat je za 10% veći od duže stranice okvira (okviri su tesni)
MAX_DOG_CUT = 0.20             # do 20% duže stranice okvira smemo da odsečemo umesto da dopunjujemo
MIN_CENTER_CROP_RATIO = 0.60   # rezervna varijanta: centralni isečak samo ako ostaje bar 60% duže strane

DOG_LABEL = "dog"
# Ostale COCO životinje i plišani meda. Kad psa nema iznad praga, detektor je psa često označio kao
# jednu od njih (čupave rase kao ovcu ili medveda, krupne kao kravu ili konja). Čovek nije na spisku.
ANIMAL_LABELS = ("cat", "sheep", "bear", "horse", "cow", "bird", "elephant", "zebra", "giraffe", "teddy bear")

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 u pikselima originala


@dataclass(frozen=True)
class Detection:
    label: str    # ime COCO klase, npr. "dog"
    score: float  # pouzdanost detektora, 0-1
    box: Box


@dataclass(frozen=True)
class Square:
    """Kvadrat u koordinatama originala. Može delom da izlazi van slike (tada se dopunjava)."""
    left: int
    top: int
    side: int

    def is_inside(self, width: int, height: int) -> bool:
        return (self.left >= 0 and self.top >= 0
                and self.left + self.side <= width and self.top + self.side <= height)


def load_image(source: str | BinaryIO) -> Image.Image:
    """Otvara sliku, okreće je po EXIF oznaci (fotografije sa telefona) i prebacuje u RGB."""
    with Image.open(source) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def select_dog(detections: list[Detection], threshold: float = DOG_SCORE_THRESHOLD) -> Detection | None:
    """Detekcija koju smatramo psom, ili None kad je nema.

    1. Najsigurniji okvir klase „dog“ iznad praga. Kad na slici ima više pasa, uzimamo jednog,
       i to onog koji je detektoru najjasniji.
    2. Ako psa nema: najsigurniji okvir druge životinje iznad praga. Na svakoj slici u našem skupu
       je pas, pa je takav okvir skoro uvek pas sa pogrešnom oznakom.
    3. Inače None, pa se koristi rezervna varijanta (ili se slika izbacuje iz train skupa).
    """
    for labels in ((DOG_LABEL,), ANIMAL_LABELS):
        found = [d for d in detections if d.label in labels and d.score >= threshold]
        if found:
            return max(found, key=lambda d: d.score)
    return None


def place_square(width: int, height: int, center_x: float, center_y: float, side: float) -> Square:
    """Postavlja kvadrat na zadati centar i pomera ga ka unutra koliko može.

    Po svakoj osi: ako kvadrat staje u sliku, ceo ostaje unutar slike. Ako ne staje,
    pokriva celu tu stranu slike, a višak (dopuna) se deli tako da centar ostane što bliže zadatom.
    """
    side = max(1, round(side))

    def axis_start(center: float, length: int) -> int:
        lowest, highest = min(0, length - side), max(0, length - side)
        return min(max(round(center - side / 2), lowest), highest)

    return Square(axis_start(center_x, width), axis_start(center_y, height), side)


def dog_square(width: int, height: int, box: Box) -> Square:
    """Kvadrat oko psa: duža stranica okvira plus margina, centriran na okvir.

    - Okvir staje u kraću stranu slike: margina se smanjuje koliko treba da ne bi bilo dopune.
      Margina tako nikad sama ne izaziva dopunu.
    - Okvir je duži od kraće strane slike, ali za najviše MAX_DOG_CUT svoje duže stranice:
      kvadrat je kraća strana slike. Odsecaju se krajevi psa (rep, šape, leđa), bez dopune.
    - Okvir je još duži: kvadrat obuhvata celog psa, a deo van slike se dopunjuje.
      Kvadrat tada ne izlazi van duže strane slike.
    """
    x1, y1, x2, y2 = box
    longer = max(x2 - x1, y2 - y1)
    short_side, long_side = min(width, height), max(width, height)
    side = longer * (1 + BOX_MARGIN)
    side = min(side, short_side) if longer * (1 - MAX_DOG_CUT) <= short_side else min(side, long_side)
    return place_square(width, height, (x1 + x2) / 2, (y1 + y2) / 2, side)


def box_cut(box: Box, square: Square) -> float:
    """Koliki deo duže stranice okvira ostaje van kvadrata (0 kad je ceo okvir u kvadratu).

    Kvadrat je centriran na okvir, pa se kod odsecanja seče podjednako sa oba kraja,
    osim kad okvir dodiruje ivicu slike.
    """
    x1, y1, x2, y2 = box
    return max(0.0, 1 - square.side / max(x2 - x1, y2 - y1))


def fallback_square(width: int, height: int) -> Square:
    """Rezervna varijanta kad psa nema: centralni kvadrat, a kod jako izduženih slika dopuna do kvadrata."""
    short_side, long_side = min(width, height), max(width, height)
    side = short_side if short_side / long_side >= MIN_CENTER_CROP_RATIO else long_side
    return place_square(width, height, width / 2, height / 2, side)


def crop_square(img: Image.Image, square: Square, fill: tuple[int, int, int]) -> Image.Image:
    """Iseca kvadrat iz originala. Deo kvadrata van slike popunjava bojom fill."""
    right, bottom = square.left + square.side, square.top + square.side
    if square.is_inside(img.width, img.height):
        return img.crop((square.left, square.top, right, bottom))
    canvas = Image.new("RGB", (square.side, square.side), fill)
    inside = img.crop((max(square.left, 0), max(square.top, 0), min(right, img.width), min(bottom, img.height)))
    canvas.paste(inside, (max(-square.left, 0), max(-square.top, 0)))
    return canvas


@dataclass(frozen=True)
class Prepared:
    image: Image.Image     # IMAGE_SIZE x IMAGE_SIZE, RGB
    dog: Detection | None  # izabrani okvir psa; None: korišćena je rezervna varijanta
    padded: bool           # True: deo slike je dopuna srednjom bojom
    cut: float             # deo duže stranice okvira koji je odsečen (0-MAX_DOG_CUT)
    square: Square

    @property
    def has_dog(self) -> bool:
        return self.dog is not None


def prepare_image(img: Image.Image, detections: list[Detection], fill: tuple[int, int, int]) -> Prepared:
    """Glavna funkcija: od originala i detekcija pravi ulaz klasifikatora (224 x 224).

    Pas se iseca iz originala u punoj rezoluciji, pa se tek onda smanjuje, da sačuva što više detalja.
    LANCZOS pri smanjenju radi antialiasing, pa krzno ne dobija talasaste šare.
    """
    dog = select_dog(detections)
    if dog is None:
        square, cut = fallback_square(img.width, img.height), 0.0
    else:
        square = dog_square(img.width, img.height, dog.box)
        cut = box_cut(dog.box, square)
    crop = crop_square(img, square, fill)
    resized = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return Prepared(resized, dog=dog, padded=not square.is_inside(img.width, img.height), cut=cut,
                    square=square)


def settings(fill: tuple[int, int, int]) -> dict:
    """Sva podešavanja pripreme. Čuvaju se u kešu i checkpoint-u, da API radi isto što i trening."""
    return {
        "image_size": IMAGE_SIZE, "fill_rgb": list(fill), "dog_score_threshold": DOG_SCORE_THRESHOLD,
        "box_margin": BOX_MARGIN, "max_dog_cut": MAX_DOG_CUT, "min_center_crop_ratio": MIN_CENTER_CROP_RATIO,
        "dog_label": DOG_LABEL, "animal_labels": list(ANIMAL_LABELS),
    }
