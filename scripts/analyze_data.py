"""Analiza Kaggle skupa „Dog Breed Identification“: rase, veličine slika, formati i duplikati.

Pokretanje iz korena repoa, sa aktiviranim .venv:
    python scripts/analyze_data.py

Ulaz:  data/raw/labels.csv i data/raw/train/*.jpg
Izlaz: rezime u terminalu i grafikoni u docs/img/
"""

import hashlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # crtamo direktno u fajl, bez prozora
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from image_hash import dhash

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
LABELS_CSV = RAW_DIR / "labels.csv"
IMG_DIR = ROOT / "docs" / "img"

TARGET_SIZE = 224  # ulazna rezolucija koju verovatno koristimo od faze 2
EXIF_ORIENTATION_TAG = 0x0112  # 1 = slika je već uspravna, ostale vrednosti traže rotaciju

# Boje grafikona: jedna serija, pa je dovoljna jedna boja. Mreža i ose su diskretne.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def check_files(labels: pd.DataFrame) -> None:
    """Proverava da li se labels.csv i fajlovi u train/ poklapaju jedan na jedan."""
    file_ids = {p.stem for p in TRAIN_DIR.glob("*.jpg")}
    label_ids = set(labels["id"])
    print(f"Redova u labels.csv:     {len(labels)}")
    print(f"Jedinstvenih id-jeva:    {len(label_ids)}")
    print(f"Fajlova u train/:        {len(file_ids)}")
    print(f"Oznaka bez fajla:        {len(label_ids - file_ids)}")
    print(f"Fajlova bez oznake:      {len(file_ids - label_ids)}")


def summarize_breeds(labels: pd.DataFrame) -> pd.Series:
    counts = labels["breed"].value_counts()
    print(f"\nBroj rasa:               {counts.size}")
    print(f"Slika po rasi:           najmanje {counts.min()}, medijana {counts.median():.0f}, "
          f"prosek {counts.mean():.1f}, najviše {counts.max()}")
    print(f"Najviše / najmanje:      {counts.max() / counts.min():.2f}x")
    print("Najmanje slika:          " + ", ".join(f"{b} ({n})" for b, n in counts.tail(5).items()))
    print("Najviše slika:           " + ", ".join(f"{b} ({n})" for b, n in counts.head(5).items()))
    return counts


def collect_image_stats(labels: pd.DataFrame) -> pd.DataFrame:
    """Za svaku sliku: veličina fajla, SHA-256, dHash, dimenzije, format, mod boja, EXIF orijentacija, greška."""
    rows = []
    for i, image_id in enumerate(labels["id"], start=1):
        data = (TRAIN_DIR / f"{image_id}.jpg").read_bytes()
        row = {"id": image_id, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "error": ""}
        try:
            with Image.open(io.BytesIO(data)) as img:
                row.update(width=img.width, height=img.height, format=img.format, mode=img.mode,
                           orientation=img.getexif().get(EXIF_ORIENTATION_TAG, 1))
                img.load()  # dekodira celu sliku: oštećen fajl tek ovde baca grešku
                row["dhash"] = dhash(img)
        except Exception as exc:  # beležimo svaku grešku, ne prekidamo analizu
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i % 2000 == 0:
            print(f"  obrađeno {i} / {len(labels)}")
    return pd.DataFrame(rows).merge(labels, on="id")


def summarize_images(stats: pd.DataFrame) -> None:
    ok = stats[stats["error"] == ""]
    short_side = ok[["width", "height"]].min(axis=1)
    aspect = ok["width"] / ok["height"]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]

    print(f"\nOštećenih / nečitljivih: {len(stats) - len(ok)}")
    for _, row in stats[stats["error"] != ""].iterrows():
        print(f"  {row['id']}: {row['error']}")
    print(f"Formati:                 {ok['format'].value_counts().to_dict()}")
    print(f"Modovi boja:             {ok['mode'].value_counts().to_dict()}")
    print(f"EXIF orijentacija != 1:  {(ok['orientation'] != 1).sum()}")
    print(f"Ukupno na disku:         {stats['bytes'].sum() / 1024**2:.0f} MiB, "
          f"medijana {stats['bytes'].median() / 1024:.0f} KiB po slici")

    print("\nPercentili (5 / 25 / 50 / 75 / 95):")
    for name, values in [("širina", ok["width"]), ("visina", ok["height"]), ("kraća strana", short_side)]:
        print(f"  {name:13s} " + " / ".join(f"{v:.0f}" for v in values.quantile(q)) +
              f"  (min {values.min()}, max {values.max()})")
    print("  širina/visina " + " / ".join(f"{v:.2f}" for v in aspect.quantile(q)))
    print(f"Kraća strana < {TARGET_SIZE} px:  {(short_side < TARGET_SIZE).sum()} "
          f"({(short_side < TARGET_SIZE).mean():.1%})")
    print(f"Položene / uspravne / kvadratne: {(aspect > 1).mean():.1%} / {(aspect < 1).mean():.1%} / "
          f"{(aspect == 1).mean():.1%}")


def print_groups(stats: pd.DataFrame, key: str, title: str) -> None:
    groups = stats.groupby(key).filter(lambda g: len(g) > 1).groupby(key)
    print(f"\n{title}: {groups.ngroups} grupa, {sum(len(g) for _, g in groups)} slika")
    for value, g in groups:
        tag = "RAZLIČITE RASE" if g["breed"].nunique() > 1 else "ista rasa"
        files = "isti fajl" if g["sha256"].nunique() == 1 else "različiti fajlovi"
        print(f"  {value[:12]}  {tag}, {files}: " + ", ".join(f"{r.id} ({r.breed})" for r in g.itertuples()))


def summarize_duplicates(stats: pd.DataFrame) -> None:
    """Duplikati: potpuno isti fajlovi (SHA-256) i ista slika drugačije sačuvana (dHash)."""
    print_groups(stats, "sha256", "Potpuno isti fajlovi (SHA-256)")
    print_groups(stats, "dhash", "Ista slika po dHash-u (uključuje i gornje)")


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_breed_counts(counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    x = range(counts.size)
    ax.bar(x, counts.values, width=0.8, color=SERIES)
    median = counts.median()
    ax.axhline(median, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.text(counts.size - 1, median + 2, f"medijana {median:.0f}", ha="right", va="bottom",
            color=TEXT_MUTED, fontsize=9)
    for idx in (0, counts.size - 1):  # natpis samo na krajnjim vrednostima
        ax.annotate(f"{counts.index[idx]} ({counts.iloc[idx]})", (idx, counts.iloc[idx]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="left" if idx == 0 else "right", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(-1, counts.size)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xlabel(f"Rase, sortirane po broju slika ({counts.size})", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Broj slika", color=TEXT_MUTED, fontsize=10)
    ax.set_title("Broj slika po rasi (train)", color=TEXT, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_image_sizes(stats: pd.DataFrame, path: Path) -> None:
    ok = stats[stats["error"] == ""]
    short_side = ok[["width", "height"]].min(axis=1)
    aspect = ok["width"] / ok["height"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150, facecolor=SURFACE)
    for ax in (ax1, ax2):
        style_axes(ax)

    max_side = short_side.quantile(0.99)  # 1% najvećih ne širi osu
    ax1.hist(short_side.clip(upper=max_side), bins=40, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    ax1.axvline(TARGET_SIZE, color=TEXT, linewidth=1, linestyle="--")
    ax1.text(TARGET_SIZE, ax1.get_ylim()[1] * 0.95, f" {TARGET_SIZE} px", color=TEXT, fontsize=9, va="top")
    ax1.set_title("Kraća strana slike (px)", color=TEXT, fontsize=11, loc="left")
    ax1.set_xlabel("poslednji stub: 1% najvećih slika", color=TEXT_MUTED, fontsize=9)
    ax1.set_ylabel("Broj slika", color=TEXT_MUTED, fontsize=10)

    ax2.hist(aspect.clip(0.4, 2.5), bins=40, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    ax2.axvline(1.0, color=TEXT, linewidth=1, linestyle="--")
    ax2.text(1.0, ax2.get_ylim()[1] * 0.95, "kvadrat ", color=TEXT, fontsize=9, va="top", ha="right")
    ax2.set_title("Odnos širina / visina", color=TEXT, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not LABELS_CSV.exists() or not TRAIN_DIR.exists():
        print(f"GREŠKA: nema podataka u {RAW_DIR}. Uputstvo za preuzimanje je u docs/01-podaci.md.")
        return 1

    labels = pd.read_csv(LABELS_CSV)
    check_files(labels)
    counts = summarize_breeds(labels)

    print("\nČitam sve slike (dekodiranje, hash)...")
    stats = collect_image_stats(labels)
    summarize_images(stats)
    summarize_duplicates(stats)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_breed_counts(counts, IMG_DIR / "01-rase.png")
    plot_image_sizes(stats, IMG_DIR / "01-velicine-slika.png")
    print(f"\nGrafikoni: {IMG_DIR / '01-rase.png'}, {IMG_DIR / '01-velicine-slika.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
