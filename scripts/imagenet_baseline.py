"""Koliko gotov ImageNet model već zna naše rase, bez ikakvog treniranja na našem skupu.

Ovo nije deo našeg sistema. Jedan broj koji pokazuje zašto ne koristimo ImageNet težine:
ImageNet-1k ima svih 120 naših rasa kao klase, a naš skup je izveden iz njega (vidi docs/01-podaci.md).

Pokretanje iz korena repoa, sa aktiviranim .venv (posle build_cache.py):
    python scripts/imagenet_baseline.py

Ulaz:  data/cache/val.npy, val.csv, info.json (isti isečci koje ocenjujemo i kod našeg modela)
Izlaz: rezime u terminalu. Težine (~45 MB) torchvision preuzima pri prvom pokretanju.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
DOG_BLOCK = range(151, 276)  # ImageNet klase pasa (151-268) i divljih pasa (269-275)
BATCH_SIZE = 256


def normalize_name(name: str) -> str:
    """„Staffordshire bullterrier“ i „staffordshire_bullterrier“ daju isto ime."""
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def imagenet_indices(breeds: list[str]) -> list[int]:
    """ImageNet indeks za svaku našu rasu, istim redom kao naši indeksi klasa.

    Tražimo samo u bloku pasa, jer „cardigan“ postoji i kao džemper (klasa 474).
    """
    categories = WEIGHTS.meta["categories"]
    by_name = {normalize_name(categories[i]): i for i in DOG_BLOCK}
    missing = [b for b in breeds if normalize_name(b) not in by_name]
    if missing:
        raise RuntimeError(f"Nema ImageNet klase za: {missing}")
    return [by_name[normalize_name(b)] for b in breeds]


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    top = logits.topk(3, dim=1).indices
    return (top[:, 0] == targets).float().mean().item(), (top == targets[:, None]).any(dim=1).float().mean().item()


@torch.inference_mode()
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    info = json.loads((CACHE_DIR / "info.json").read_text(encoding="utf-8"))
    breeds = info["breeds"]
    ours_to_imagenet = torch.tensor(imagenet_indices(breeds))

    device = torch.device("cuda")
    images = torch.from_numpy(np.load(CACHE_DIR / "val.npy")).to(device)
    labels = torch.tensor(pd.read_csv(CACHE_DIR / "val.csv")["label"].to_numpy(), device=device)
    # ImageNet normalizacija, jer je model tako treniran (ne naša srednja vrednost i std)
    mean = torch.tensor(WEIGHTS.transforms().mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(WEIGHTS.transforms().std, device=device).view(1, 3, 1, 1)

    model = resnet18(weights=WEIGHTS).eval().to(device)  # originalna glava sa 1000 klasa, FP32
    logits = torch.cat([
        model((images[i:i + BATCH_SIZE].permute(0, 3, 1, 2).float() / 255 - mean) / std)
        for i in range(0, len(images), BATCH_SIZE)
    ]).float()

    idx = ours_to_imagenet.to(device)
    top1_120, top3_120 = accuracy(logits[:, idx], labels)   # samo naših 120 klasa, naši indeksi
    top1_1000, top3_1000 = accuracy(logits, idx[labels])     # svih 1000 klasa
    print(f"ImageNet ResNet-18 ({WEIGHTS}), bez treninga na našem skupu, val: {len(labels)} slika")
    print(f"  izbor između naših 120 rasa:  top-1 {top1_120:.4f}  top-3 {top3_120:.4f}")
    print(f"  izbor između svih 1000 klasa: top-1 {top1_1000:.4f}  top-3 {top3_1000:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
