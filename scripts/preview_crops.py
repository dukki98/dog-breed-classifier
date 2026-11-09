"""Kontakt listovi za vizuelnu proveru detektora i isečaka (samo train i val, test ne gledamo).

Pokretanje iz korena repoa, sa aktiviranim .venv (posle build_cache.py):
    python scripts/preview_crops.py

Izlaz: PNG listovi u data/previews/. Ne idu u git, jer sadrže slike iz skupa, a licenca ne dozvoljava
njihovo objavljivanje.

Svaka pločica: levo original sa okvirima (crveno: izabrani okvir psa, žuto: ostali psi iznad praga,
plavo: kvadrat koji se iseca, sivo: najsigurnija detekcija kad okvira nema), desno ulaz klasifikatora.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from preprocess import (DOG_LABEL, DOG_SCORE_THRESHOLD, IMAGE_SIZE, Detection, load_image, prepare_image)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "train"
DETECTIONS_CSV = ROOT / "data" / "detections.csv"
CACHE_DIR = ROOT / "data" / "cache"
SPLITS_DIR = ROOT / "splits"
OUT_DIR = ROOT / "data" / "previews"
SPLITS = ("train", "val")
COLUMNS, MAX_TILES = 4, 24
CAPTION = 22
BACKGROUND = (40, 40, 40)
FONT = ImageFont.load_default(size=14)


def tile(image_id: str, breed: str, dets: list[Detection], fill: tuple[int, int, int]) -> Image.Image:
    img = load_image(IMAGE_DIR / f"{image_id}.jpg")
    prepared = prepare_image(img, dets, fill)
    scale = IMAGE_SIZE / max(img.size)
    thumb = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    left = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    left.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(left)

    def rect(box, color, width=2):
        draw.rectangle([v * scale for v in box], outline=color, width=width)

    chosen = prepared.dog
    dogs = [d for d in dets if d.label == DOG_LABEL and d.score >= DOG_SCORE_THRESHOLD]
    for d in dogs:
        if d != chosen:
            rect(d.box, (255, 210, 0))
    top = max(dets, key=lambda d: d.score, default=None)
    if chosen is None and top is not None:
        rect(top.box, (170, 170, 170))
    if chosen is not None:
        rect(chosen.box, (230, 40, 40), 3)
    sq = prepared.square
    rect((sq.left, sq.top, sq.left + sq.side, sq.top + sq.side), (40, 140, 255))

    out = Image.new("RGB", (2 * IMAGE_SIZE + 4, IMAGE_SIZE + CAPTION), BACKGROUND)
    out.paste(left, (0, 0))
    out.paste(prepared.image, (IMAGE_SIZE + 4, 0))
    best_dog = max((d.score for d in dets if d.label == DOG_LABEL), default=0.0)
    if chosen is None:
        note = f"bez okvira; {top.label} {top.score:.2f}" if top else "bez ijedne detekcije"
    elif chosen.label != DOG_LABEL:
        note = f"okvir: {chosen.label} {chosen.score:.2f}, pas {best_dog:.2f}"
    else:
        note = f"pas {best_dog:.2f}, pasa {len(dogs)}"
    if prepared.padded:
        note += ", dopuna"
    if prepared.cut > 0:
        note += f", odsecanje {prepared.cut:.0%}"  # PIL font nema „č“
    ImageDraw.Draw(out).text((4, IMAGE_SIZE + 3), f"{breed[:20]} | {note}", fill=(235, 235, 235), font=FONT)
    return out


def sheet(rows: pd.DataFrame, detections: dict[str, list[Detection]], fill, path: Path) -> None:
    rows = rows.head(MAX_TILES)
    if rows.empty:
        return
    tiles = [tile(r.id, r.breed, detections.get(r.id, []), fill) for r in rows.itertuples()]
    w, h = tiles[0].size
    n_rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    canvas = Image.new("RGB", (COLUMNS * (w + 6), n_rows * (h + 6)), (255, 255, 255))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % COLUMNS) * (w + 6), (i // COLUMNS) * (h + 6)))
    canvas.save(path)
    print(f"  {path.name}: {len(tiles)} pločica")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    info = json.loads((CACHE_DIR / "info.json").read_text(encoding="utf-8"))
    fill = tuple(info["preprocess"]["fill_rgb"])
    dets_df = pd.read_csv(DETECTIONS_CSV)
    detections: dict[str, list[Detection]] = {}
    for r in dets_df.dropna(subset=["label"]).itertuples(index=False):
        detections.setdefault(r.id, []).append(Detection(r.label, r.score, (r.x1, r.y1, r.x2, r.y2)))

    # Oznake iz keša. Slike bez okvira koje su izbačene iz train skupa nemaju red u kešu.
    images = pd.concat([pd.read_csv(SPLITS_DIR / f"{s}.csv").assign(split=s) for s in SPLITS])
    flags = pd.concat([pd.read_csv(CACHE_DIR / f"{s}.csv", keep_default_na=False) for s in SPLITS])
    images = images.merge(flags[["id", "has_dog", "box_class", "padded", "cut"]], on="id", how="left")
    images = images.fillna({"has_dog": False, "box_class": "", "padded": False, "cut": 0.0})
    images = images.astype({"has_dog": bool, "padded": bool})
    images["dog_score"] = images["id"].map(
        lambda i: max((d.score for d in detections.get(i, []) if d.label == DOG_LABEL), default=0.0))
    images["n_dogs"] = images["id"].map(
        lambda i: sum(d.label == DOG_LABEL and d.score >= DOG_SCORE_THRESHOLD for d in detections.get(i, [])))
    is_dog = images["box_class"] == DOG_LABEL

    def shuffled(rows: pd.DataFrame) -> pd.DataFrame:
        return rows.sample(frac=1, random_state=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Listovi u {OUT_DIR}:")
    sheet(shuffled(images[images["has_dog"] & ~is_dog]), detections, fill, OUT_DIR / "1-okvir-zivotinje.png")
    sheet(shuffled(images[~images["has_dog"]]), detections, fill, OUT_DIR / "2-bez-okvira.png")
    sheet(images[is_dog].sort_values("dog_score"), detections, fill, OUT_DIR / "3-najnesigurniji-psi.png")
    sheet(shuffled(images[images["n_dogs"] >= 2]), detections, fill, OUT_DIR / "4-vise-pasa.png")
    sheet(shuffled(images[images["has_dog"] & images["padded"]]), detections, fill, OUT_DIR / "5-dopuna.png")
    sheet(images.sort_values("cut", ascending=False), detections, fill, OUT_DIR / "6-najvise-odseceno.png")
    sheet(shuffled(images[images["has_dog"]]), detections, fill, OUT_DIR / "7-nasumicno.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
