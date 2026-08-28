from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hamming_distance_hex(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hashes must have the same length.")
    return sum(
        bin(int(a, 16) ^ int(b, 16)).count("1") for a, b in zip(left, right, strict=True)
    )


def image_average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pillow to compute perceptual image hashes.") from exc

    with Image.open(path) as image:
        grayscale = image.convert("L").resize((hash_size, hash_size))
        pixels = list(grayscale.getdata())

    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"
