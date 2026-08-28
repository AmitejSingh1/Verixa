"""WildFake (ModelScope) ingestion via cached metadata CSV + targeted per-file image download.

Design:
- Uses the ModelScope cached train_metadata.csv to select exact rows for target architectures
  and real images, eliminating sequential streaming over millions of unrelated rows.
- Uses ModelScope HubApi.get_dataset_file_url to fetch signed download URLs for only the
  selected samples.
- Normalizes all images to 224×224 JPEGs (Q=90) and produces a 7-column Manifest CSV.
- Preserves the Architecture metadata in the `generator` column for synthetic media.
"""

from __future__ import annotations

import csv
import io
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from verixa.data.schema import MANIFEST_COLUMNS, ManifestRow, manifest_path_to_str
from verixa.utils.hashing import sha256_file
from verixa.utils.images import save_pil_jpeg

_COL_ARCHITECTURE = "Architecture"
_COL_IS_FAKE = "IsFake"
_COL_IMAGE_PATH = "Image_path"

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "datasets" / "downloads"


def find_cached_train_metadata_csv() -> Path | None:
    """Locate the cached train_metadata.csv (size ~311 MB) in the modelscope cache."""
    if not _DEFAULT_CACHE_DIR.exists():
        return None
    for file_path in _DEFAULT_CACHE_DIR.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size == 311894734:
            return file_path
    return None


def sample_wildfake_rows(
    metadata_csv: Path,
    caps_per_architecture: dict[str, int],
    real_cap: int,
    seed: int = 1337,
) -> list[dict[str, str]]:
    """Sample targeted rows from train_metadata.csv stratified by architecture and real/fake."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, str]]] = {"__REAL__": []}
    for arch in caps_per_architecture:
        buckets[arch] = []

    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            is_fake = row.get(_COL_IS_FAKE, "").strip()
            arch = row.get(_COL_ARCHITECTURE, "").strip()
            if is_fake == "0":
                buckets["__REAL__"].append(row)
            elif is_fake == "1" and arch in caps_per_architecture:
                buckets[arch].append(row)

    selected: list[dict[str, str]] = []

    # Sample real rows
    real_pool = buckets["__REAL__"]
    rng.shuffle(real_pool)
    selected.extend(real_pool[:real_cap])

    # Sample fake rows per architecture
    for arch, cap in caps_per_architecture.items():
        pool = buckets.get(arch, [])
        rng.shuffle(pool)
        selected.extend(pool[:cap])

    # Shuffle the combined selected list
    rng.shuffle(selected)
    return selected


def ingest_wildfake(
    dataset_name: str = "hy2628982280/WildFake",
    metadata_csv: Path | None = None,
    output_root: Path = Path("data/processed/wildfake"),
    manifest_path: Path = Path("data/manifests/wildfake_manifest.csv"),
    caps_per_architecture: dict[str, int] | None = None,
    real_cap: int = 1000,
    seed: int = 1337,
    jpeg_quality: int = 90,
    download_retries: int = 3,
    download_timeout: int = 30,
) -> dict[str, Any]:
    """Ingest a controlled WildFake subset by sampling metadata and downloading target images."""
    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise RuntimeError("Install modelscope before ingesting WildFake.") from exc

    if caps_per_architecture is None:
        caps_per_architecture = {
            "DDPM": 400,
            "BigGAN": 300,
            "ADM": 200,
            "DALLE": 100,
            "Midjourney": 100,
        }

    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_csv is None:
        metadata_csv = find_cached_train_metadata_csv()
        if metadata_csv is None:
            raise FileNotFoundError(
                "Could not auto-locate cached train_metadata.csv in ~/.cache/modelscope/."
            )

    namespace, name = dataset_name.split("/", maxsplit=1)
    api = HubApi()

    selected_rows = sample_wildfake_rows(
        metadata_csv=metadata_csv,
        caps_per_architecture=caps_per_architecture,
        real_cap=real_cap,
        seed=seed,
    )

    rows: list[ManifestRow] = []
    counts: Counter[str] = Counter()
    corrupt = 0
    download_errors = 0
    written = 0

    progress = tqdm(total=len(selected_rows), desc="ingest WildFake")
    try:
        for row in selected_rows:
            is_fake = int(row.get(_COL_IS_FAKE, 1))
            arch = row.get(_COL_ARCHITECTURE, "").strip()
            rel_path = row.get(_COL_IMAGE_PATH, "").strip().lstrip("./")

            label = is_fake  # 0 = REAL, 1 = AI-GENERATED

            image_bytes = _download_image(
                api=api,
                file_name=rel_path,
                dataset_name=name,
                namespace=namespace,
                retries=download_retries,
                timeout=download_timeout,
            )
            if image_bytes is None:
                download_errors += 1
                progress.update(1)
                continue

            arch_slug = arch.lower().replace("-", "_").replace(" ", "_") if label == 1 else "real"
            subdir = "real" if label == 0 else f"fake_{arch_slug}"
            destination = output_root / subdir / f"wildfake_{written:07d}.jpg"

            try:
                _save_bytes_as_jpeg(image_bytes, destination, size=(224, 224), quality=jpeg_quality)
            except Exception:
                corrupt += 1
                progress.update(1)
                continue

            digest = sha256_file(destination)
            rows.append(
                ManifestRow(
                    image_path=manifest_path_to_str(destination),
                    label=label,
                    source_dataset="WildFake",
                    split="unassigned",
                    generator=arch if label == 1 else None,
                    original_id=rel_path,
                    sha256=digest,
                )
            )
            counts[f"{'real' if label == 0 else arch}:{label}"] += 1
            written += 1
            progress.update(1)
    finally:
        progress.close()

    _write_manifest(manifest_path, rows)

    return {
        "dataset": dataset_name,
        "source_dataset": "WildFake",
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "target_rows": len(selected_rows),
        "written_rows": written,
        "corrupt_or_unreadable": corrupt,
        "download_errors": download_errors,
        "class_architecture_counts": dict(sorted(counts.items())),
        "seed": seed,
        "jpeg_quality": jpeg_quality,
        "disk_usage_mb": _directory_size_mb(output_root),
    }


def _download_image(
    api: Any,
    file_name: str,
    dataset_name: str,
    namespace: str,
    retries: int,
    timeout: int,
) -> bytes | None:
    for attempt in range(retries):
        try:
            url: str = api.get_dataset_file_url(
                file_name=file_name,
                dataset_name=dataset_name,
                namespace=namespace,
                revision="master",
            )
            req = urllib.request.Request(url, headers={"User-Agent": "python-modelscope/verixa"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(2**attempt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(2**attempt)
    return None


def _save_bytes_as_jpeg(
    image_bytes: bytes,
    destination_path: Path,
    size: tuple[int, int],
    quality: int,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pillow.") from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as img:
        save_pil_jpeg(img, destination_path, size=size, quality=quality)


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _directory_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / 1024**2, 3)
