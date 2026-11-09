"""Pravi keš ulaza za klasifikator: za svaki skup jedan uint8 niz slika 224 x 224 u .npy fajlu.

Pokretanje iz korena repoa, sa aktiviranim .venv (posle detect_dogs.py):
    python scripts/build_cache.py

Ulaz:  data/raw/train/*.jpg, data/detections.csv, splits/*.csv
Izlaz u data/cache/:
    {skup}.npy   slike, uint8, N x 224 x 224 x 3 (visina, širina, kanal)
    {skup}.csv   id, breed, label, has_dog, box_class, padded, cut (isti redosled kao u .npy)
    info.json    rase (redosled je indeks klase), podešavanja pripreme (i boja dopune) i brojevi

Train: slike na kojima detektor nije našao ni psa ni drugu životinju se izbacuju.
Val i test: ne izbacuje se nijedna slika, nego se koristi rezervna varijanta, kao u produkciji.
JPEG se dekodira samo ovde, jednom. Trening posle čita gotove nizove.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import preprocess
from preprocess import Detection, load_image, prepare_image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "train"
DETECTIONS_CSV = ROOT / "data" / "detections.csv"
SPLITS_DIR = ROOT / "splits"
CACHE_DIR = ROOT / "data" / "cache"
SPLITS = ("train", "val", "test")
WORKERS = os.cpu_count() or 4  # Pillow pušta GIL pri dekodiranju i smanjenju, pa niti stvarno rade paralelno


def load_detections() -> dict[str, list[Detection]]:
    dets = pd.read_csv(DETECTIONS_CSV).dropna(subset=["label"])
    result: dict[str, list[Detection]] = {}
    for r in dets.itertuples(index=False):
        result.setdefault(r.id, []).append(Detection(r.label, r.score, (r.x1, r.y1, r.x2, r.y2)))
    return result


def mean_color(ids: list[str]) -> tuple[int, int, int]:
    """Srednja boja train skupa: prosek srednjih boja slika (svaka slika ima istu težinu).

    Računa se samo iz train skupa, da val i test ni na koji način ne utiču na pripremu.
    """
    def image_mean(image_id: str) -> np.ndarray:
        return np.asarray(load_image(IMAGE_DIR / f"{image_id}.jpg"), dtype=np.float64).mean(axis=(0, 1))

    with ThreadPoolExecutor(WORKERS) as pool:
        means = np.stack(list(pool.map(image_mean, ids)))
    return tuple(int(round(v)) for v in means.mean(axis=0))


def build_split(df: pd.DataFrame, detections: dict[str, list[Detection]], fill: tuple[int, int, int],
                drop_without_dog: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Priprema sve slike jednog skupa. Slike se upisuju direktno u jedan unapred zauzet niz,
    da u RAM-u ne bi istovremeno postojale i pojedinačne slike i ceo niz."""
    def prepare(image_id: str):
        img = load_image(IMAGE_DIR / f"{image_id}.jpg")
        return prepare_image(img, detections.get(image_id, []), fill)

    size = preprocess.IMAGE_SIZE
    images = np.empty((len(df), size, size, 3), dtype=np.uint8)
    flags, kept = [], 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for p in pool.map(prepare, df["id"]):
            flags.append({"has_dog": p.has_dog, "box_class": p.dog.label if p.dog else "",
                          "padded": p.padded, "cut": round(p.cut, 4)})
            if p.has_dog or not drop_without_dog:
                images[kept] = np.asarray(p.image)
                kept += 1

    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    if drop_without_dog:
        df = df[df["has_dog"]]
    return images[:kept], df.reset_index(drop=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not DETECTIONS_CSV.exists():
        print(f"GREŠKA: nema {DETECTIONS_CSV}. Prvo pokreni scripts/detect_dogs.py.")
        return 1

    splits = {s: pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in SPLITS}
    breeds = sorted(splits["train"]["breed"].unique())
    label_of = {b: i for i, b in enumerate(breeds)}
    detections = load_detections()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    fill = mean_color(list(splits["train"]["id"]))
    print(f"Srednja boja train skupa (RGB): {fill}  ({time.perf_counter() - start:.0f} s)")

    counts = {}
    for split, df in splits.items():
        t = time.perf_counter()
        images, meta = build_split(df, detections, fill, drop_without_dog=(split == "train"))
        meta = meta.assign(label=meta["breed"].map(label_of))
        np.save(CACHE_DIR / f"{split}.npy", images)
        meta[["id", "breed", "label", "has_dog", "box_class", "padded", "cut"]].to_csv(
            CACHE_DIR / f"{split}.csv", index=False, lineterminator="\n")
        has_dog = meta["has_dog"]
        counts[split] = {"images": len(meta), "dropped_without_dog": len(df) - len(meta),
                         "fallback": int((~has_dog).sum()),
                         "animal_box": int((has_dog & (meta["box_class"] != preprocess.DOG_LABEL)).sum()),
                         "padded": int((has_dog & meta["padded"]).sum()), "cut": int((meta["cut"] > 0).sum())}
        c = counts[split]
        print(f"{split:5s}: {c['images']} slika, izbačeno bez psa {c['dropped_without_dog']}, "
              f"rezervna varijanta {c['fallback']}, okvir druge životinje {c['animal_box']}, "
              f"pas sa dopunom {c['padded']}, pas odsečen {c['cut']}, "
              f"{images.nbytes / 1024**3:.2f} GiB, {time.perf_counter() - t:.0f} s")

    info = {"breeds": breeds, "preprocess": preprocess.settings(fill), "counts": counts}
    (CACHE_DIR / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Ukupno {time.perf_counter() - start:.0f} s. Keš je u {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
