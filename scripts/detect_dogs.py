"""Pokreće gotov COCO detektor na svim slikama iz podele i čuva sve njegove detekcije.

Pokretanje iz korena repoa, sa aktiviranim .venv:
    python scripts/detect_dogs.py

Ulaz:  splits/train.csv, val.csv, test.csv i data/raw/train/*.jpg
Izlaz: data/detections.csv, jedan red po detekciji: id, width, height, label, score, x1, y1, x2, y2.
       Slika bez ijedne detekcije ima jedan red sa praznim label.

Čuvamo sve detekcije (ne samo izabrani okvir psa), da bi prag i pravila isecanja mogli da se menjaju
bez ponovnog pokretanja detektora. Težine (~74 MB) torchvision preuzima pri prvom pokretanju.
"""

import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.transforms.functional import pil_to_tensor

from preprocess import Detection, load_image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "train"
SPLITS_DIR = ROOT / "splits"
OUT_CSV = ROOT / "data" / "detections.csv"

WEIGHTS = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
CATEGORIES = WEIGHTS.meta["categories"]  # 91 COCO imena, indeks = broj klase koji vraća model
BATCH_SIZE = 8


def load_detector(device: torch.device) -> torch.nn.Module:
    """Faster R-CNN MobileNetV3-Large FPN sa COCO težinama, bez doučavanja, u režimu za inference.

    Model sam menja veličinu slike (kraća strana 800 px) i normalizuje je, pa dobija sliku u opsegu 0-1.
    Vraća detekcije sa pouzdanošću >= 0,05 (podrazumevano u torchvision-u), najviše 100 po slici.
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=WEIGHTS)
    return model.eval().to(device)


@torch.inference_mode()
def detect(model: torch.nn.Module, images: list[torch.Tensor], device: torch.device) -> list[list[Detection]]:
    """Detekcije za listu slika (uint8 tenzori 3 x H x W). Okviri su u pikselima originala."""
    batch = [img.to(device, non_blocking=True).float().div_(255) for img in images]
    results = []
    for out in model(batch):
        results.append([
            Detection(CATEGORIES[label], float(score), tuple(float(v) for v in box))
            for label, score, box in zip(out["labels"].tolist(), out["scores"].tolist(), out["boxes"].tolist())
        ])
    return results


class ImageFiles(Dataset):
    """Učitava slike u pozadinskim procesima, dok GPU radi detekciju na prethodnom batch-u."""

    def __init__(self, ids: list[str]):
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[str, torch.Tensor]:
        image_id = self.ids[i]
        return image_id, pil_to_tensor(load_image(IMAGE_DIR / f"{image_id}.jpg"))


def collate(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], list[torch.Tensor]]:
    """Slike su različitih veličina, pa ih ne slažemo u jedan tenzor, već vraćamo listu."""
    ids, images = zip(*batch)
    return list(ids), list(images)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not IMAGE_DIR.exists():
        print(f"GREŠKA: nema slika u {IMAGE_DIR}. Uputstvo je u docs/01-podaci.md.")
        return 1
    if not torch.cuda.is_available():
        print("GREŠKA: CUDA nije dostupna.")
        return 1

    ids = sorted(pd.concat([pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")])["id"])
    device = torch.device("cuda")
    model = load_detector(device)
    loader = DataLoader(ImageFiles(ids), batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate,
                        pin_memory=True)

    print(f"Detekcija na {len(ids)} slika...")
    start = time.perf_counter()
    rows = 0
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "width", "height", "label", "score", "x1", "y1", "x2", "y2"])
        for batch_index, (batch_ids, images) in enumerate(loader, start=1):
            for image_id, img, dets in zip(batch_ids, images, detect(model, images, device)):
                height, width = img.shape[1:]
                if not dets:
                    writer.writerow([image_id, width, height, "", "", "", "", "", ""])
                for d in dets:
                    writer.writerow([image_id, width, height, d.label, f"{d.score:.4f}",
                                     *(f"{v:.1f}" for v in d.box)])
                rows += max(1, len(dets))
            if batch_index * BATCH_SIZE % 2000 < BATCH_SIZE:
                print(f"  obrađeno {min(batch_index * BATCH_SIZE, len(ids))} / {len(ids)}")
    elapsed = time.perf_counter() - start
    print(f"Gotovo za {elapsed:.0f} s ({len(ids) / elapsed:.1f} slika/s), {rows} redova u {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
