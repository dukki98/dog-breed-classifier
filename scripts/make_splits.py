"""Podela skupa na train / val / test (70 / 15 / 15), stratifikovano po rasi, bez duplikata.

Pokretanje iz korena repoa, sa aktiviranim .venv:
    python scripts/make_splits.py

Ulaz:  data/raw/labels.csv i data/raw/train/*.jpg
Izlaz: splits/train.csv, splits/val.csv, splits/test.csv (kolone id, breed)
       splits/excluded.csv (izbačene slike i razlog)

Fajlovi u splits/ idu u git, da bi svi eksperimenti koristili potpuno istu podelu.
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from image_hash import dhash

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
LABELS_CSV = RAW_DIR / "labels.csv"
SPLITS_DIR = ROOT / "splits"

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def compute_hashes(labels: pd.DataFrame) -> pd.DataFrame:
    """Dodaje dHash i broj piksela svakoj slici."""
    rows = []
    for i, image_id in enumerate(labels["id"], start=1):
        with Image.open(TRAIN_DIR / f"{image_id}.jpg") as img:
            rows.append({"id": image_id, "dhash": dhash(img), "pixels": img.width * img.height})
        if i % 2000 == 0:
            print(f"  obrađeno {i} / {len(labels)}")
    return labels.merge(pd.DataFrame(rows), on="id")


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vraća (zadržane, izbačene). Grupa = slike sa istim dHash-om (ista fotografija).

    - Grupa sa različitim rasama: izbacuju se sve slike, jer nema jedne tačne oznake.
    - Grupa sa istom rasom: ostaje slika sa najviše piksela (kod istog broja, manji id),
      ostale se izbacuju. Tako ista fotografija ne može da bude i u train i u test skupu.
    """
    group_size = df.groupby("dhash")["id"].transform("size")
    conflict = df.groupby("dhash")["breed"].transform("nunique") > 1

    ranked = df.sort_values(["dhash", "pixels", "id"], ascending=[True, False, True])
    is_best_copy = ~ranked["dhash"].duplicated()  # prva u grupi posle sortiranja
    extra_copy = (group_size > 1) & ~conflict & ~is_best_copy

    excluded = pd.concat([
        df[conflict].assign(reason="ista slika, različite rase"),
        df[extra_copy].assign(reason="kopija iste slike, ista rasa"),
    ])
    kept = df[~conflict & ~extra_copy]
    return kept, excluded


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stratifikovana podela u dva koraka: prvo train / ostatak, pa ostatak na val / test."""
    df = df.sort_values("id")  # podela ne sme da zavisi od redosleda ulaza
    holdout = VAL_FRACTION + TEST_FRACTION
    train, rest = train_test_split(df, test_size=holdout, stratify=df["breed"], random_state=SEED)
    val, test = train_test_split(rest, test_size=TEST_FRACTION / holdout, stratify=rest["breed"],
                                 random_state=SEED)
    return {"train": train, "val": val, "test": test}


def check_splits(splits: dict[str, pd.DataFrame], n_breeds: int) -> None:
    """Prekida rad ako podela nije ispravna (namerno raise, a ne assert, koji može biti isključen)."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for column in ("id", "dhash"):
                shared = set(splits[a][column]) & set(splits[b][column])
                if shared:
                    raise RuntimeError(f"{a} i {b} dele {len(shared)} vrednosti kolone {column}")
    for name, s in splits.items():
        if s["breed"].nunique() != n_breeds:
            raise RuntimeError(f"{name} ima {s['breed'].nunique()} rasa umesto {n_breeds}")


def report(splits: dict[str, pd.DataFrame], excluded: pd.DataFrame, total: int) -> None:
    print(f"\nIzbačeno: {len(excluded)} od {total}")
    print(excluded["reason"].value_counts().to_string())
    kept_total = sum(len(s) for s in splits.values())
    print(f"\n{'skup':6s} {'slika':>6s} {'udeo':>6s} {'rasa':>5s}   po rasi: min / medijana / max")
    for name, s in splits.items():
        per_breed = s["breed"].value_counts()
        print(f"{name:6s} {len(s):6d} {len(s) / kept_total:6.1%} {per_breed.size:5d}   "
              f"{per_breed.min()} / {per_breed.median():.0f} / {per_breed.max()}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not LABELS_CSV.exists() or not TRAIN_DIR.exists():
        print(f"GREŠKA: nema podataka u {RAW_DIR}. Uputstvo za preuzimanje je u docs/01-podaci.md.")
        return 1

    labels = pd.read_csv(LABELS_CSV)
    print("Računam dHash za sve slike...")
    df = compute_hashes(labels)
    kept, excluded = remove_duplicates(df)
    splits = split(kept)
    check_splits(splits, n_breeds=labels["breed"].nunique())
    report(splits, excluded, total=len(labels))

    SPLITS_DIR.mkdir(exist_ok=True)
    for name, s in splits.items():
        s.sort_values("id")[["id", "breed"]].to_csv(SPLITS_DIR / f"{name}.csv", index=False, lineterminator="\n")
    excluded.sort_values(["reason", "dhash", "id"])[["id", "breed", "reason", "dhash"]].to_csv(
        SPLITS_DIR / "excluded.csv", index=False, lineterminator="\n")
    print(f"\nUpisano u {SPLITS_DIR}: train.csv, val.csv, test.csv, excluded.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
