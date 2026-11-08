"""Perceptualni hash slike, za pronalaženje iste fotografije sačuvane na drugačiji način."""

import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 8) -> str:
    """dHash: da li je svaki piksel svetliji od desnog suseda, na slici smanjenoj na 9x8 u sivom.

    Ista fotografija sačuvana sa drugom kompresijom ili u drugoj veličini ima drugačiji
    SHA-256, ali najčešće isti dHash. Vraća 64 bita kao 16 heksadecimalnih cifara.
    """
    small = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, :-1] > small[:, 1:]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
