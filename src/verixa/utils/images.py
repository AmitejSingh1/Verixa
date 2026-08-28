from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def probe_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        return {"valid": False, "error": f"Install pillow first: {exc}"}

    try:
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
        return {"valid": True, "width": width, "height": height, "format": fmt}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def resize_and_save_jpeg(
    source_path: Path,
    destination_path: Path,
    size: tuple[int, int] = (224, 224),
    quality: int = 90,
) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install pillow before ingesting images.") from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        save_pil_jpeg(image, destination_path, size=size, quality=quality)


def save_pil_jpeg(
    image: object,
    destination_path: Path,
    size: tuple[int, int] = (224, 224),
    quality: int = 90,
) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install pillow before ingesting images.") from exc

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image).__name__}.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized = normalized.resize(size, Image.Resampling.BICUBIC)
    normalized.save(destination_path, format="JPEG", quality=quality, optimize=True)
