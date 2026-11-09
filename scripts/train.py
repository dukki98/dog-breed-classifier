"""Trening klasifikatora rasa od nule: ResNet-18 bez ImageNet težina, na kešu iz build_cache.py.

Pokretanje iz korena repoa, sa aktiviranim .venv, dok je laptop na punjaču:
    python scripts/train.py --seed 42

Ulaz:  data/cache/train.npy, val.npy, train.csv, val.csv, info.json
Izlaz: runs/<ime>/log.csv           jedan red po epohi (loss, tačnost, stopa učenja, trajanje)
       runs/<ime>/summary.json      konačni rezultat, trajanje, VRAM
       runs/<ime>/val_top3.csv      3 najverovatnije rase za svaku val sliku (za analizu grešaka)
       checkpoints/<ime>.pt         model posle poslednje epohe i sve što treba za pripremu ulaza

Ceo train i val skup stoje na GPU-u kao uint8 (~1,2 GiB), pa nema učitavanja sa diska tokom treninga.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"
CHECKPOINT_DIR = ROOT / "checkpoints"
EVAL_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trening ResNet-18 od nule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="početna stopa učenja (kosinusno opada do 0)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--min-scale", type=float, default=0.5, help="najmanji isečak, kao udeo površine slike")
    p.add_argument("--name", default=None, help="ime pokretanja (podrazumevano resnet18-s<seed>)")
    return p.parse_args()


def load_split(split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Slike (uint8, N x 224 x 224 x 3) i oznake jednog skupa, odmah na GPU-u."""
    images = torch.from_numpy(np.load(CACHE_DIR / f"{split}.npy")).to(device)
    meta = pd.read_csv(CACHE_DIR / f"{split}.csv")
    labels = torch.tensor(meta["label"].to_numpy(), dtype=torch.long, device=device)
    return images, labels, meta


def to_float(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 N x H x W x C u float N x C x H x W, opseg 0-1.

    permute ne kopira podatke, pa rezultat ima raspored channels_last (kanal je poslednji u memoriji),
    što tensor jezgra koriste za konvolucije bez preuređivanja.
    """
    return images_u8.permute(0, 3, 1, 2).float().div_(255)


@torch.no_grad()
def channel_stats(images_u8: torch.Tensor, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Srednja vrednost i standardna devijacija po kanalu za ceo train skup (po delovima, da stane u VRAM)."""
    total = torch.zeros(3, dtype=torch.float64, device=images_u8.device)
    total_sq = torch.zeros_like(total)
    for i in range(0, len(images_u8), chunk):
        x = images_u8[i:i + chunk].double().div_(255)
        total += x.sum(dim=(0, 1, 2))
        total_sq += x.square().sum(dim=(0, 1, 2))
    count = images_u8.shape[0] * images_u8.shape[1] * images_u8.shape[2]
    mean = total / count
    std = (total_sq / count - mean.square()).sqrt()
    return mean.float(), std.float()


def random_crop_flip(x: torch.Tensor, min_scale: float) -> torch.Tensor:
    """Augmentacija na GPU-u: svaka slika u batch-u dobija svoj nasumični isečak i ogledanje.

    torchvision-ove transformacije na celom batch-u primenjuju isti nasumični isečak na sve slike,
    pa ovde to radimo sami. Za svaku sliku pravimo afinu transformaciju (theta) koja kaže koji deo
    ulaza se preslikava u izlaz:
    - površina isečka je min_scale do 100% slike, odnos stranica 3/4 do 4/3;
    - položaj isečka je nasumičan, ali isečak uvek ostaje unutar slike;
    - u pola slučajeva se slika ogleda levo-desno (predznak na x osi).
    grid_sample zatim bilinearno uzorkuje isečak i vraća ga na punu veličinu (224 x 224).
    """
    n, device = x.shape[0], x.device
    area = torch.empty(n, device=device).uniform_(min_scale, 1.0)
    log_ratio = torch.empty(n, device=device).uniform_(math.log(3 / 4), math.log(4 / 3))
    ratio = log_ratio.exp()
    # Širina i visina isečka u odnosu na celu sliku (1 = cela strana)
    w = (area * ratio).sqrt().clamp(max=1.0)
    h = (area / ratio).sqrt().clamp(max=1.0)
    # Centar isečka u normalizovanim koordinatama (-1 do 1), tako da isečak ne izađe iz slike
    cx = (torch.rand(n, device=device) * 2 - 1) * (1 - w)
    cy = (torch.rand(n, device=device) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)

    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, images_u8: torch.Tensor, labels: torch.Tensor, mean: torch.Tensor,
             std: torch.Tensor) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Loss, top-1 i top-3 tačnost na celom skupu, bez augmentacije. Vraća i top-3 predikcije."""
    model.eval()
    loss_sum, top1, top3 = 0.0, 0, 0
    all_idx, all_prob = [], []
    for i in range(0, len(images_u8), EVAL_BATCH_SIZE):
        x = (to_float(images_u8[i:i + EVAL_BATCH_SIZE]) - mean) / std
        y = labels[i:i + EVAL_BATCH_SIZE]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        prob, idx = logits.softmax(dim=1).topk(3, dim=1)
        top1 += (idx[:, 0] == y).sum().item()
        top3 += (idx == y[:, None]).any(dim=1).sum().item()
        all_idx.append(idx)
        all_prob.append(prob)
    n = len(images_u8)
    metrics = {"loss": loss_sum / n, "top1": top1 / n, "top3": top3 / n}
    return metrics, torch.cat(all_idx).cpu(), torch.cat(all_prob).cpu()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    name = args.name or f"resnet18-s{args.seed}"
    if not (CACHE_DIR / "info.json").exists():
        print(f"GREŠKA: nema keša u {CACHE_DIR}. Prvo pokreni scripts/build_cache.py.")
        return 1

    torch.manual_seed(args.seed)
    # cudnn bira najbrži algoritam za konvolucije. Zbog toga dva pokretanja sa istim seed-om
    # nisu identična do poslednjeg bita, ali su statistički ista.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    info = json.loads((CACHE_DIR / "info.json").read_text(encoding="utf-8"))
    x_train, y_train, _ = load_split("train", device)
    x_val, y_val, val_meta = load_split("val", device)
    mean, std = channel_stats(x_train)
    mean_b, std_b = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    print(f"train {len(x_train)}, val {len(x_val)} slika; mean {mean.tolist()}, std {std.tolist()}")

    model = resnet18(weights=None, num_classes=len(info["breeds"]))  # weights=None: nasumične težine
    model = model.to(device, memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())

    steps_per_epoch = len(x_train) // args.batch_size  # poslednji nepun batch se preskače
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Kosinusni raspored: stopa učenja glatko opada od lr do 0 tokom celog treninga (menja se posle svakog koraka)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_rows = []
    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        lr = scheduler.get_last_lr()[0]
        perm = torch.randperm(len(x_train), device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            x = random_crop_flip(to_float(x_train[idx]), args.min_scale)
            x = ((x - mean_b) / std_b).contiguous(memory_format=torch.channels_last)
            y = y_train[idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Zbir ostaje na GPU-u: .item() u svakom koraku bi čekao GPU i usporio trening
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(dim=1) == y).sum()
        torch.cuda.synchronize()
        train_time = time.perf_counter() - epoch_start
        seen = steps_per_epoch * args.batch_size

        val, _, _ = evaluate(model, x_val, y_val, mean_b, std_b)
        row = {"epoch": epoch, "lr": lr, "train_loss": loss_sum.item() / seen, "train_top1": correct.item() / seen,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top3": val["top3"],
               "train_seconds": train_time, "epoch_seconds": time.perf_counter() - epoch_start}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "log.csv", index=False, lineterminator="\n")
        print(f"epoha {epoch:3d}  lr {lr:.4f}  train loss {row['train_loss']:.3f} top1 {row['train_top1']:.3f}  "
              f"val loss {val['loss']:.3f} top1 {val['top1']:.3f} top3 {val['top3']:.3f}  "
              f"{row['epoch_seconds']:.1f} s",
              flush=True)

    total_time = time.perf_counter() - train_start
    final, top_idx, top_prob = evaluate(model, x_val, y_val, mean_b, std_b)
    breeds = info["breeds"]
    pd.DataFrame({
        "id": val_meta["id"], "breed": val_meta["breed"], "has_dog": val_meta["has_dog"],
        "box_class": val_meta["box_class"],
        **{f"pred{k + 1}": [breeds[i] for i in top_idx[:, k].tolist()] for k in range(3)},
        **{f"prob{k + 1}": top_prob[:, k].numpy().round(4) for k in range(3)},
    }).to_csv(run_dir / "val_top3.csv", index=False, lineterminator="\n")

    log = pd.DataFrame(log_rows)
    best = log.loc[log["val_top1"].idxmax()]
    summary = {
        "name": name, "args": vars(args), "params": n_params,
        "train_images": len(x_train), "val_images": len(x_val),
        "final_val": final,
        "best_val_top1": {"epoch": int(best["epoch"]), "top1": float(best["val_top1"]),
                          "top3": float(best["val_top3"])},
        "total_minutes": total_time / 60,
        "mean_train_seconds_per_epoch": float(log["train_seconds"].mean()),
        "train_images_per_second": steps_per_epoch * args.batch_size / float(log["train_seconds"].mean()),
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "torchvision": torchvision.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")

    torch.save({
        "model": model.state_dict(), "arch": "resnet18", "breeds": breeds,
        # Sve što API mora da uradi isto kao trening: priprema slike i normalizacija
        "preprocess": info["preprocess"],
        "norm_mean": mean.tolist(), "norm_std": std.tolist(), "args": vars(args),
    }, CHECKPOINT_DIR / f"{name}.pt")

    print(f"\nGotovo za {total_time / 60:.1f} min. Val top-1 {final['top1']:.4f}, top-3 {final['top3']:.4f}. "
          f"Najveći VRAM {summary['peak_vram_reserved_gib']:.2f} GiB (rezervisano).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
