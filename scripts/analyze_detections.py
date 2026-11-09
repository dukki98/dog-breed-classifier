"""Analiza detekcija: koliko slika ima psa iznad praga, koliko pasa po slici, šta detektor vidi kad
psa ne nađe i koliko isečaka traži dopunu.

Pokretanje iz korena repoa, sa aktiviranim .venv (posle detect_dogs.py):
    python scripts/analyze_detections.py

Ulaz:  data/detections.csv, splits/train.csv i val.csv (test skup ne gledamo)
Izlaz: rezime u terminalu i grafikon docs/img/02-prag-detektora.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # crtamo direktno u fajl, bez prozora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import (DOG_LABEL, DOG_SCORE_THRESHOLD, IMAGE_SIZE, MAX_DOG_CUT, Detection, box_cut, dog_square,
                        select_dog)

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_CSV = ROOT / "data" / "detections.csv"
SPLITS_DIR = ROOT / "splits"
IMG_DIR = ROOT / "docs" / "img"
# Test skup ne gledamo do kraja projekta: prag i pravila isecanja biramo samo na osnovu train i val.
SPLITS = ("train", "val")
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)

# Isti stil kao grafikoni iz faze 1
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def per_image(dets: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Jedan red po slici: najsigurniji pas, broj pasa iznad praga, najsigurnija detekcija bilo koje klase
    i okvir koji pravila iz preprocess.py biraju (pas, druga životinja ili ništa)."""
    images = dets.groupby("id")[["width", "height"]].first()
    dogs = dets[dets["label"] == DOG_LABEL]
    images["dog_score"] = dogs.groupby("id")["score"].max()
    images["dog_score"] = images["dog_score"].fillna(0.0)
    images["n_dogs"] = dogs[dogs["score"] >= threshold].groupby("id").size()
    images["n_dogs"] = images["n_dogs"].fillna(0).astype(int)
    top = dets.dropna(subset=["label"]).sort_values("score", ascending=False).drop_duplicates("id")
    images = images.join(top.set_index("id")[["label", "score"]].rename(
        columns={"label": "top_label", "score": "top_score"}))

    chosen = {}
    for image_id, g in dets.dropna(subset=["label"]).groupby("id"):
        d = select_dog([Detection(r.label, r.score, (r.x1, r.y1, r.x2, r.y2)) for r in g.itertuples()], threshold)
        if d is not None:
            chosen[image_id] = (d.label, d.score, *d.box)
    columns = ["box_class", "box_score", "x1", "y1", "x2", "y2"]
    chosen = pd.DataFrame.from_dict(chosen, orient="index", columns=columns)
    images = images.join(chosen)
    images["has_dog"] = images["box_class"].notna()
    return images


def crop_stats(images: pd.DataFrame) -> pd.DataFrame:
    """Za slike sa izabranim okvirom: udeo okvira u slici, dopuna, odsecanje i stranica kvadrata."""
    rows = []
    for image_id, r in images[images["has_dog"]].iterrows():
        box = (r.x1, r.y1, r.x2, r.y2)
        square = dog_square(int(r.width), int(r.height), box)
        rows.append({"id": image_id,
                     "box_share": (r.x2 - r.x1) * (r.y2 - r.y1) / (r.width * r.height),
                     "padded": not square.is_inside(int(r.width), int(r.height)),
                     "cut": box_cut(box, square),
                     "side": square.side})
    return pd.DataFrame(rows).set_index("id")


def report(images: pd.DataFrame, threshold: float) -> None:
    q = [0.01, 0.05, 0.25, 0.5]
    print(f"Prag: {threshold}\n")
    print(f"{'skup':6s} {'slika':>6s}  " + "  ".join(f"pas>={t:.1f}" for t in THRESHOLDS)
          + "   najsigurniji pas, percentili 1 / 5 / 25 / 50")
    for split in SPLITS:
        s = images[images["split"] == split]
        counts = "  ".join(f"{(s['dog_score'] >= t).mean():8.1%}" for t in THRESHOLDS)
        pct = " / ".join(f"{v:.3f}" for v in s["dog_score"].quantile(q))
        print(f"{split:6s} {len(s):6d}  {counts}   {pct}")

    no_dog = images[images["dog_score"] < threshold]
    print(f"\nBez psa iznad praga {threshold}: {len(no_dog)} ({len(no_dog) / len(images):.1%})")
    print("  najsigurnija detekcija na tim slikama: "
          + ", ".join(f"{k} ({v})" for k, v in no_dog["top_label"].fillna("ništa").value_counts().items()))

    animal = images[images["has_dog"] & (images["box_class"] != DOG_LABEL)]
    print(f"\nOkvir druge životinje umesto psa: {len(animal)} ("
          + ", ".join(f"{k} {v}" for k, v in animal["box_class"].value_counts().items()) + ")")
    for split in SPLITS:
        s = images[images["split"] == split]
        print(f"  {split}: okvir ima {s['has_dog'].mean():.1%} slika, bez okvira {(~s['has_dog']).sum()}")

    by_breed = images.groupby("breed").agg(no_dog=("dog_score", lambda s: (s < threshold).mean()),
                                           no_box=("has_dog", lambda s: (~s).mean()), n=("has_dog", "size"))
    print("  rase sa najviše slika bez psa: udeo bez psa -> udeo bez ikakvog okvira")
    for breed, r in by_breed.sort_values("no_dog", ascending=False).head(8).iterrows():
        print(f"    {breed:32s} {r.no_dog:6.1%} -> {r.no_box:5.1%}  (od {int(r.n)} slika)")

    print("\nBroj pasa iznad praga po slici (sve slike):")
    print("  " + ", ".join(f"{k}: {v}" for k, v in images["n_dogs"].value_counts().sort_index().items()))

    crops = crop_stats(images)
    cut = crops[crops["cut"] > 0]
    print(f"\nIsečci oko izabranog okvira ({len(crops)} slika):")
    print(f"  dopuna srednjom bojom: {crops['padded'].sum()} ({crops['padded'].mean():.1%})")
    print(f"  odsecanje (do {MAX_DOG_CUT:.0%} duže stranice okvira): {len(cut)} ({len(cut) / len(crops):.1%}), "
          "odsečeno, percentili 25 / 50 / 75: "
          + " / ".join(f"{v:.1%}" for v in cut["cut"].quantile([0.25, 0.5, 0.75])))
    print("  udeo okvira u površini slike, percentili 5 / 25 / 50 / 75 / 95: "
          + " / ".join(f"{v:.0%}" for v in crops["box_share"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])))
    print(f"  stranica kvadrata manja od {IMAGE_SIZE} px (isečak se uvećava): "
          f"{(crops['side'] < IMAGE_SIZE).sum()} ({(crops['side'] < IMAGE_SIZE).mean():.1%})")
    print("  stranica kvadrata u px, percentili 5 / 25 / 50 / 75 / 95: "
          + " / ".join(f"{v:.0f}" for v in crops["side"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])))


def plot_threshold(images: pd.DataFrame, threshold: float, path: Path) -> None:
    """Udeo slika na kojima je pas nađen, u zavisnosti od praga pouzdanosti."""
    t = np.linspace(0, 1, 201)
    share = [(images["dog_score"] >= x).mean() * 100 for x in t]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.plot(t, share, color=SERIES, linewidth=2)
    at = (images["dog_score"] >= threshold).mean() * 100
    ax.axvline(threshold, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"prag {threshold}: {at:.1f}%", (threshold, at), xytext=(6, -14), textcoords="offset points",
                color=TEXT, fontsize=9)
    low = max(0, min(share) - 2)
    ax.set_ylim(low, 100.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prag pouzdanosti detektora za klasu „dog“", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Slika sa psom (%)", color=TEXT_MUTED, fontsize=10)
    ax.set_title(f"Na koliko slika je pas nađen (train i val, {len(images)} slika)", color=TEXT, fontsize=12,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not DETECTIONS_CSV.exists():
        print(f"GREŠKA: nema {DETECTIONS_CSV}. Prvo pokreni scripts/detect_dogs.py.")
        return 1
    dets = pd.read_csv(DETECTIONS_CSV)
    split_of = pd.concat([pd.read_csv(SPLITS_DIR / f"{s}.csv").assign(split=s) for s in SPLITS])
    images = per_image(dets, DOG_SCORE_THRESHOLD)
    images = images.join(split_of.set_index("id")[["split", "breed"]], how="inner")
    report(images, DOG_SCORE_THRESHOLD)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_threshold(images, DOG_SCORE_THRESHOLD, IMG_DIR / "02-prag-detektora.png")
    print(f"\nGrafikon: {IMG_DIR / '02-prag-detektora.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
