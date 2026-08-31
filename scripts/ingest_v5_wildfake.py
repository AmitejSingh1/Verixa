"""Targeted ingestion of 5,000 WildFake images for Hybrid V5 development experiment.

Datasets ingested:
- 2,500 DALL-E 2 images from Typical/DALLE2/
- 1,250 CelebA-HQ real photography images
- 1,250 AFHQ real animal photography images

Exclusions strictly enforced:
- Zero DALL-E 3 / Advanced images
- Zero COCO images
- Zero images from benchmark_dataset/
"""
from __future__ import annotations

import csv
import io
import random
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi
from PIL import Image
from tqdm import tqdm

from verixa.data.schema import MANIFEST_COLUMNS, ManifestRow, manifest_path_to_str
from verixa.utils.hashing import sha256_file
from verixa.utils.images import save_pil_jpeg


class RemoteZipStream(io.RawIOBase):
    """Virtual seekable stream that fetches zip file chunks over HTTP Range requests."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        size = len(b)
        if size == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + size - 1, self.size - 1)
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "python-modelscope/verixa", "Range": f"bytes={self.pos}-{end}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def get_remote_zip_file(api: HubApi, zip_rel_path: str) -> tuple[zipfile.ZipFile, str]:
    url = api.get_dataset_file_url(
        file_name=zip_rel_path,
        dataset_name="WildFake",
        namespace="hy2628982280",
        revision="master",
    )
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "python-modelscope/verixa"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
    stream = io.BufferedReader(RemoteZipStream(url, total_size), buffer_size=256 * 1024)
    zf = zipfile.ZipFile(stream)
    return zf, url


def load_prohibited_dalle3_list(api: HubApi) -> set[str]:
    """Fetch dalle3.csv to obtain the full set of 8,843 prohibited benchmark filenames."""
    url = api.get_dataset_file_url(
        file_name="label_csv_files/dalle3.csv",
        dataset_name="WildFake",
        namespace="hy2628982280",
        revision="master",
    )
    req = urllib.request.Request(url, headers={"User-Agent": "python-modelscope/verixa"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    prohibited = set()
    for row in reader:
        p = row.get("Image_path", "").strip()
        if p:
            fname = Path(p).name
            prohibited.add(fname)
            prohibited.add(p)
    return prohibited


def fetch_and_save_image(
    zf: zipfile.ZipFile,
    entry_name: str,
    out_path: Path,
    target_size: tuple[int, int] = (224, 224),
    quality: int = 90,
) -> bool:
    try:
        with zf.open(entry_name) as f:
            raw_bytes = f.read()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            save_pil_jpeg(img, out_path, size=target_size, quality=quality)
        return True
    except Exception as e:
        print(f"Error processing {entry_name}: {e}", file=sys.stderr)
        return False


def main() -> int:
    print("=================================================================")
    print(" Verixa — Ingesting Controlled WildFake Subset for Hybrid V5")
    print("=================================================================")
    api = HubApi()
    random.seed(1337)

    out_root = Path("data/processed/wildfake")
    dalle2_dir = out_root / "dalle2"
    celeba_dir = out_root / "celebahq"
    afhq_dir = out_root / "afhq"

    dalle2_dir.mkdir(parents=True, exist_ok=True)
    celeba_dir.mkdir(parents=True, exist_ok=True)
    afhq_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch prohibited DALL-E 3 list
    print("Fetching authoritative prohibited benchmark list (dalle3.csv)...")
    dalle3_prohibited = load_prohibited_dalle3_list(api)
    print(f"Loaded {len(dalle3_prohibited):,} prohibited benchmark references.")

    # 2. Ingest DALL-E 2 (2,500 images)
    print("\nConnecting to DALLE.zip remote central directory...")
    zf_dalle, _ = get_remote_zip_file(api, "Images/Diffusion_based/DALLE.zip")
    all_dalle_names = zf_dalle.namelist()
    typical_dalle2 = [
        n for n in all_dalle_names
        if "typical/dalle2" in n.lower() and n.lower().endswith((".jpg", ".png"))
    ]
    print(f"Discovered {len(typical_dalle2):,} available Typical DALL-E 2 images.")

    # Verification: Ensure zero intersection with dalle3 prohibited
    intersection = [
        n for n in typical_dalle2
        if Path(n).name in dalle3_prohibited or n in dalle3_prohibited
    ]
    assert len(intersection) == 0, (
        f"Critical contamination detected: {len(intersection)} items overlap!"
    )
    print(f"Exclusion verification: {len(intersection)} benchmark items overlap (100% CLEAN).")

    # Deterministically sample 2,500
    random.seed(1337)
    sampled_dalle2 = sorted(random.sample(typical_dalle2, 2500))

    print(f"Extracting & resizing 2,500 DALL-E 2 images to {dalle2_dir}...")
    dalle2_records: list[tuple[Path, str]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for idx, entry_name in enumerate(sampled_dalle2):
            dest = dalle2_dir / f"wildfake_dalle2_{idx:05d}.jpg"
            fut = ex.submit(fetch_and_save_image, zf_dalle, entry_name, dest)
            futures[fut] = (dest, entry_name)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="DALL-E 2"):
            dest, orig = futures[fut]
            if fut.result():
                dalle2_records.append((dest, orig))

    print(f"DALL-E 2 completed: {len(dalle2_records)} images in {time.perf_counter() - t0:.1f}s.")

    # 3. Ingest CelebA-HQ (1,250 images)
    print("\nConnecting to celebahq.zip remote central directory...")
    zf_celeba, _ = get_remote_zip_file(api, "Images/Real/celebahq.zip")
    celeba_names = [
        n for n in zf_celeba.namelist()
        if n.lower().endswith((".jpg", ".png"))
    ]
    print(f"Discovered {len(celeba_names):,} available CelebA-HQ images.")
    random.seed(1337)
    sampled_celeba = sorted(random.sample(celeba_names, 1250))

    celeba_records: list[tuple[Path, str]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for idx, entry_name in enumerate(sampled_celeba):
            dest = celeba_dir / f"wildfake_celeba_{idx:05d}.jpg"
            fut = ex.submit(fetch_and_save_image, zf_celeba, entry_name, dest)
            futures[fut] = (dest, entry_name)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="CelebA-HQ"):
            dest, orig = futures[fut]
            if fut.result():
                celeba_records.append((dest, orig))

    print(f"CelebA-HQ completed: {len(celeba_records)} images in {time.perf_counter() - t0:.1f}s.")

    # 4. Ingest AFHQ (1,250 images)
    print("\nConnecting to afhq.zip remote central directory...")
    zf_afhq, _ = get_remote_zip_file(api, "Images/Real/afhq.zip")
    afhq_names = [
        n for n in zf_afhq.namelist()
        if n.lower().endswith((".jpg", ".png"))
    ]
    print(f"Discovered {len(afhq_names):,} available AFHQ images.")
    random.seed(1337)
    sampled_afhq = sorted(random.sample(afhq_names, 1250))

    afhq_records: list[tuple[Path, str]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for idx, entry_name in enumerate(sampled_afhq):
            dest = afhq_dir / f"wildfake_afhq_{idx:05d}.jpg"
            fut = ex.submit(fetch_and_save_image, zf_afhq, entry_name, dest)
            futures[fut] = (dest, entry_name)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="AFHQ"):
            dest, orig = futures[fut]
            if fut.result():
                afhq_records.append((dest, orig))

    print(f"AFHQ completed: {len(afhq_records)} images in {time.perf_counter() - t0:.1f}s.")

    # 5. Build Manifest rows
    print("\nComputing SHA-256 hashes and compiling manifest rows...")
    new_manifest_rows: list[ManifestRow] = []

    # DALL-E 2 rows (label=1, synthetic)
    for path, orig in dalle2_records:
        new_manifest_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(path),
                label=1,
                source_dataset="WildFake",
                split="train",
                generator="DALLE2",
                original_id=orig,
                sha256=sha256_file(path),
            )
        )

    # CelebA-HQ rows (label=0, authentic)
    for path, orig in celeba_records:
        new_manifest_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(path),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=orig,
                sha256=sha256_file(path),
            )
        )

    # AFHQ rows (label=0, authentic)
    for path, orig in afhq_records:
        new_manifest_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(path),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=orig,
                sha256=sha256_file(path),
            )
        )

    print(
        f"Compiled {len(new_manifest_rows):,} new training rows (2,500 synthetic, 2,500 authentic)."
    )

    # 6. Load existing 30K manifest
    original_manifest_path = Path("data/manifests/merged_manifest.csv")
    existing_rows: list[dict[str, str]] = []
    with open(original_manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.append(row)

    orig_train = [r for r in existing_rows if r["split"] == "train"]
    orig_val = [r for r in existing_rows if r["split"] == "val"]
    print(f"Original manifest: {len(orig_train):,} train, {len(orig_val):,} val.")
    assert len(orig_val) == 6001, f"Validation set altered! Expected 6001, got {len(orig_val)}"

    # 7. Merge into v5_augmented_manifest.csv
    v5_manifest_path = Path("data/manifests/v5_augmented_manifest.csv")
    all_v5_dicts = list(existing_rows)
    for r in new_manifest_rows:
        all_v5_dicts.append(r.to_csv_dict())

    # Verify zero overlap between new images and original validation set
    val_sha256 = {r["sha256"] for r in orig_val}
    new_sha256 = {r.sha256 for r in new_manifest_rows}
    val_overlap = val_sha256.intersection(new_sha256)
    assert len(val_overlap) == 0, (
        f"Leakage detected: {len(val_overlap)} new images match validation!"
    )

    # Write out v5 manifest
    with open(v5_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_v5_dicts)

    v5_train_count = sum(1 for r in all_v5_dicts if r["split"] == "train")
    v5_val_count = sum(1 for r in all_v5_dicts if r["split"] == "val")
    print("\n=================================================================")
    print(" V5 Manifest Created Successfully!")
    print("=================================================================")
    print(f" Destination Path:    {v5_manifest_path}")
    print(f" Total Samples:       {len(all_v5_dicts):,}")
    print(f" Training Samples:    {v5_train_count:,} (23,999 orig + 5,000 new)")
    print(f" Validation Samples:  {v5_val_count:,} (Strictly identical 6,001)")
    print(" Benchmark Overlap:   EXACTLY 0 (Verified against dalle3.csv)")
    print(" Validation Overlap:  EXACTLY 0 (Verified via SHA-256)")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
