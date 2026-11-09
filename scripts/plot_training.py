"""Krive učenja i rezime za više pokretanja istog treninga (različiti seed-ovi).

Pokretanje iz korena repoa, sa aktiviranim .venv (posle train.py):
    python scripts/plot_training.py
    python scripts/plot_training.py --runs resnet18-s42 resnet18-s43 --out docs/img/moj-grafikon.png

Ulaz:  runs/<ime>/log.csv i summary.json
Izlaz: tabela u terminalu i grafikon (podrazumevano docs/img/02-krive-ucenja.png)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # crtamo direktno u fajl, bez prozora
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

# Isti stil kao grafikoni iz faze 1. Boja prati veličinu (train, val top-1, val top-3) u oba panela.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
TRAIN, VAL, VAL_TOP3 = "#2a78d6", "#eb6834", "#1baf7a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Krive učenja za više seed-ova")
    p.add_argument("--runs", nargs="+", default=["resnet18-s42", "resnet18-s43", "resnet18-s44"])
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "02-krive-ucenja.png")
    return p.parse_args()


def style(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel("Epoha", color=TEXT_MUTED, fontsize=10)


def plot(logs: dict[str, pd.DataFrame], path: Path) -> None:
    """Svaki seed je jedna tanka linija. Linije iste boje su ponavljanja istog treninga, pa se vidi rasipanje."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), dpi=150, facecolor=SURFACE)
    style(ax_loss, "Loss (cross-entropy)")
    style(ax_acc, "Tačnost (%)")
    series = [
        (ax_loss, "train_loss", TRAIN, "train", 1),
        (ax_loss, "val_loss", VAL, "val", 1),
        (ax_acc, "train_top1", TRAIN, "train top-1", 100),
        (ax_acc, "val_top1", VAL, "val top-1", 100),
        (ax_acc, "val_top3", VAL_TOP3, "val top-3", 100),
    ]
    for ax, column, color, label, scale in series:
        for i, log in enumerate(logs.values()):
            ax.plot(log["epoch"], log[column] * scale, color=color, linewidth=1.2, label=label if i == 0 else None)
        # Direktna oznaka na kraju: prosek poslednje epohe preko seed-ova
        last = sum(log[column].iloc[-1] for log in logs.values()) / len(logs) * scale
        epochs = max(log["epoch"].iloc[-1] for log in logs.values())
        ax.annotate(f"{label} {last:.2f}" if scale == 1 else f"{label} {last:.1f}", (epochs, last),
                    xytext=(4, 0), textcoords="offset points", va="center", color=TEXT, fontsize=8)
    ax_acc.set_ylim(0, 100)
    ax_loss.set_ylim(bottom=0)
    for ax in (ax_loss, ax_acc):
        ax.margins(x=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_MUTED, loc="best")
    fig.suptitle(f"ResNet-18 od nule, {len(logs)} seed-a (train metrike su sa augmentacijom)",
                 color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    logs, rows = {}, []
    for name in args.runs:
        run = RUNS_DIR / name
        logs[name] = pd.read_csv(run / "log.csv")
        s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        rows.append({"run": name, "val_top1": s["final_val"]["top1"], "val_top3": s["final_val"]["top3"],
                     "val_loss": s["final_val"]["loss"], "train_top1": logs[name]["train_top1"].iloc[-1],
                     "best_epoch": s["best_val_top1"]["epoch"], "best_top1": s["best_val_top1"]["top1"],
                     "minutes": s["total_minutes"], "s_per_epoch": s["mean_train_seconds_per_epoch"],
                     "img_per_s": s["train_images_per_second"], "vram_gib": s["peak_vram_reserved_gib"]})
    table = pd.DataFrame(rows).set_index("run")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table)
    for column in ("val_top1", "val_top3", "val_loss", "train_top1"):
        print(f"{column}: prosek {table[column].mean():.4f}, std {table[column].std():.4f} "
              f"(min {table[column].min():.4f}, max {table[column].max():.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(logs, args.out)
    print(f"Grafikon: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
