"""Provera GPU-a: da li ga PyTorch vidi i koliko brzo množi matrice u FP32, TF32 i BF16.

Pokretanje iz korena repoa, sa aktiviranim .venv:
    python scripts/check_gpu.py
"""

import platform
import sys
import time

import torch

MATRIX_SIZE = 4096  # množimo dve matrice 4096 x 4096
WARMUP_RUNS = 3  # prva pokretanja su sporija (inicijalizacija, cuBLAS bira algoritam)
TIMED_RUNS = 20  # merimo prosek više pokretanja, ne jedno


def print_environment() -> None:
    print(f"Python:              {platform.python_version()}")
    print(f"PyTorch:             {torch.__version__}")
    print(f"CUDA (build PyTorch): {torch.version.cuda}")
    print(f"cuDNN:               {torch.backends.cudnn.version()}")


def print_gpu_info() -> None:
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU:                 {props.name}")
    print(f"Compute capability:  {props.major}.{props.minor}")
    print(f"Broj SM jedinica:    {props.multi_processor_count}")
    print(f"VRAM ukupno:         {total_bytes / 1024**3:.2f} GiB")
    print(f"VRAM slobodno:       {free_bytes / 1024**3:.2f} GiB")
    print(f"BF16 podrška:        {'da' if torch.cuda.is_bf16_supported() else 'ne'}")


def matmul_tflops(dtype: torch.dtype) -> float:
    """Meri brzinu množenja matrica na GPU-u, u TFLOPS (10^12 operacija u sekundi)."""
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda", dtype=dtype)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda", dtype=dtype)

    for _ in range(WARMUP_RUNS):
        torch.matmul(a, b)

    # GPU radi asinhrono: Python samo zakaže posao i odmah ide dalje.
    # synchronize() čeka da GPU završi sve zakazano. Bez njega bismo izmerili
    # samo koliko traje zakazivanje, a ne samo računanje.
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_RUNS):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    seconds_per_run = (time.perf_counter() - start) / TIMED_RUNS

    # Množenje dve N x N matrice: N^3 množenja + N^3 sabiranja = 2 * N^3 operacija.
    operations = 2 * MATRIX_SIZE**3
    return operations / seconds_per_run / 1e12


def run_benchmarks() -> None:
    print(f"\nMnoženje matrica {MATRIX_SIZE}x{MATRIX_SIZE} (prosek {TIMED_RUNS} pokretanja):")

    # "highest": pravi FP32, računa se na običnim CUDA jezgrima.
    torch.set_float32_matmul_precision("highest")
    fp32 = matmul_tflops(torch.float32)

    # "high": ulaz je FP32, ali množenje ide na tensor jezgra u formatu TF32
    # (mantisa od 10 bitova umesto 23, isti opseg kao FP32).
    torch.set_float32_matmul_precision("high")
    tf32 = matmul_tflops(torch.float32)
    torch.set_float32_matmul_precision("highest")

    # BF16 tenzori idu direktno na tensor jezgra.
    bf16 = matmul_tflops(torch.bfloat16)

    for name, tflops in [("FP32", fp32), ("TF32", tf32), ("BF16", bf16)]:
        print(f"  {name}: {tflops:6.2f} TFLOPS  ({tflops / fp32:.1f}x u odnosu na FP32)")


def main() -> int:
    # Windows: da bi slova š, ž, č, ć, đ radila i kad se izlaz preusmeri u fajl.
    sys.stdout.reconfigure(encoding="utf-8")

    print_environment()
    if not torch.cuda.is_available():
        print("\nGREŠKA: PyTorch ne vidi CUDA GPU.")
        print("Najčešći uzrok: instalirana je CPU verzija PyTorch-a (verzija se završava na '+cpu').")
        print("Rešenje: python -m pip install -r requirements-torch.txt --force-reinstall")
        return 1

    print_gpu_info()
    run_benchmarks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
