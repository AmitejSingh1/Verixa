"""High-performance direct HTTP Range extractor for 2,500 DALL-E 2 images.

Extracts strictly from Typical/DALLE2/ inside DALLE.zip via zero-seek direct byte ranges.
Verifies zero overlap with dalle3.csv.
Resizes to 224x224 JPEGs (Q=90) and builds data/manifests/v5_augmented_manifest.csv.
"""
from __future__ import annotations

import csv
import io
import random
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modelscope.hub.api import HubApi
from PIL import Image
from tqdm import tqdm

from verixa.data.schema import MANIFEST_COLUMNS, ManifestRow, manifest_path_to_str
from verixa.utils.hashing import sha256_file
from verixa.utils.images import save_pil_jpeg


class RemoteZipStream(io.RawIOBase):
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
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def fetch_single_dalle2_image(
    url: str,
    info: zipfile.ZipInfo,
    dest_path: Path,
    retries: int = 3,
) -> bool:
    for attempt in range(retries):
        try:
            # 1. Read local header (128 bytes)
            hdr_req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "python-modelscope/verixa",
                    "Range": f"bytes={info.header_offset}-{info.header_offset + 128}",
                },
            )
            with urllib.request.urlopen(hdr_req, timeout=20) as resp:
                hdr = resp.read()

            magic, _, _, comp, _, _, _, cs, _, fl, el = struct.unpack(
                "<IHHHHHIIIHH", hdr[:30]
            )
            if magic != 0x04034B50:
                return False

            data_start = info.header_offset + 30 + fl + el
            data_end = data_start + cs - 1

            # 2. Read compressed image bytes
            data_req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "python-modelscope/verixa",
                    "Range": f"bytes={data_start}-{data_end}",
                },
            )
            with urllib.request.urlopen(data_req, timeout=20) as resp:
                cdata = resp.read()

            raw = zlib.decompress(cdata, -15) if comp == 8 else cdata
            with Image.open(io.BytesIO(raw)) as img:
                save_pil_jpeg(img, dest_path, size=(224, 224), quality=90)
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)
    return False


def main() -> int:
    print("=================================================================")
    print(" Verixa — Parallel Zero-Seek Ingestion for DALL-E 2 (V5)")
    print("=================================================================")
    api = HubApi()

    # 1. Fetch dalle3 prohibited list
    print("Verifying prohibited benchmark filenames from dalle3.csv...")
    d3_url = api.get_dataset_file_url(
        file_name="label_csv_files/dalle3.csv",
        dataset_name="WildFake",
        namespace="hy2628982280",
        revision="master",
    )
    req = urllib.request.Request(d3_url, headers={"User-Agent": "python-modelscope/verixa"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        content = resp.read().decode("utf-8", errors="ignore")

    prohibited_dalle3 = set()
    for row in csv.DictReader(io.StringIO(content)):
        p = row.get("Image_path", "").strip()
        if p:
            prohibited_dalle3.add(Path(p).name)
            prohibited_dalle3.add(p)
    print(f"Loaded {len(prohibited_dalle3):,} prohibited benchmark references.")

    # 2. Inspect DALLE.zip central directory
    dalle_url = api.get_dataset_file_url(
        file_name="Images/Diffusion_based/DALLE.zip",
        dataset_name="WildFake",
        namespace="hy2628982280",
        revision="master",
    )
    req = urllib.request.Request(
        dalle_url, method="HEAD", headers={"User-Agent": "python-modelscope/verixa"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        total_size = int(resp.headers.get("Content-Length", 25587709291))

    print(f"Connecting to DALLE.zip central directory ({total_size / (1024*1024):.1f} MB)...")
    stream = io.BufferedReader(RemoteZipStream(dalle_url, total_size), buffer_size=256 * 1024)
    with zipfile.ZipFile(stream) as zf:
        typical_entries = [
            i for i in zf.infolist()
            if "typical/dalle2" in i.filename.lower()
            and i.filename.lower().endswith((".jpg", ".png"))
        ]

    print(f"Discovered {len(typical_entries):,} typical DALL-E 2 entries.")

    # Assert 0 benchmark overlap
    intersection = [
        i for i in typical_entries
        if Path(i.filename).name in prohibited_dalle3 or i.filename in prohibited_dalle3
    ]
    assert len(intersection) == 0, f"Benchmark leakage! {len(intersection)} overlap!"
    print(f"Exclusion verification: {len(intersection)} benchmark items overlap (100% CLEAN).")

    # Sample 2,500 deterministically
    random.seed(1337)
    sampled_entries = sorted(random.sample(typical_entries, 2500), key=lambda x: x.header_offset)

    dalle2_dir = Path("data/processed/wildfake/dalle2")
    dalle2_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(sampled_entries):,} DALL-E 2 images to {dalle2_dir}...")
    t0 = time.perf_counter()
    extracted_dalle2: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {}
        for idx, entry in enumerate(sampled_entries):
            dest = dalle2_dir / f"wildfake_dalle2_{idx:05d}.jpg"
            fut = ex.submit(fetch_single_dalle2_image, dalle_url, entry, dest)
            futures[fut] = (dest, entry.filename)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="DALL-E 2 Extractor"):
            dest, orig_name = futures[fut]
            if fut.result():
                extracted_dalle2.append((dest, orig_name))

    elapsed = time.perf_counter() - t0
    rate = len(extracted_dalle2) / max(0.1, elapsed)
    print(
        f"DALL-E 2 complete: {len(extracted_dalle2):,} images in {elapsed:.1f}s ({rate:.1f} img/s)."
    )
    assert len(extracted_dalle2) == 2500, f"Expected 2500, got {len(extracted_dalle2)}"

    # 3. Verify CelebA and AFHQ images exist
    celeba_files = sorted(Path("data/processed/wildfake/celebahq").glob("*.jpg"))
    afhq_files = sorted(Path("data/processed/wildfake/afhq").glob("*.jpg"))
    print(f"CelebA-HQ files on disk: {len(celeba_files):,}")
    print(f"AFHQ files on disk:      {len(afhq_files):,}")
    assert len(celeba_files) == 1250, f"Expected 1250 CelebA, got {len(celeba_files)}"
    assert len(afhq_files) == 1250, f"Expected 1250 AFHQ, got {len(afhq_files)}"

    # 4. Build manifest rows
    print("\nComputing SHA-256 digests and assembling V5 manifest rows...")
    new_rows: list[ManifestRow] = []

    for path, orig in extracted_dalle2:
        new_rows.append(
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

    for path in celeba_files:
        new_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(path),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=f"celebahq/{path.name}",
                sha256=sha256_file(path),
            )
        )

    for path in afhq_files:
        new_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(path),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=f"afhq/{path.name}",
                sha256=sha256_file(path),
            )
        )

    print(f"Compiled {len(new_rows):,} new training rows (2,500 DALL-E 2, 2,500 Real).")

    # 5. Load original 30K manifest
    orig_rows: list[dict[str, str]] = []
    with open("data/manifests/merged_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            orig_rows.append(r)

    orig_train = [r for r in orig_rows if r["split"] == "train"]
    orig_val = [r for r in orig_rows if r["split"] == "val"]
    print(f"Original manifest: {len(orig_train):,} train, {len(orig_val):,} val.")
    assert len(orig_val) == 6001, f"Validation set changed! Got {len(orig_val)}"

    # 6. Verify zero leakage with validation set
    val_hashes = {r["sha256"] for r in orig_val}
    new_hashes = {r.sha256 for r in new_rows}
    leakage = val_hashes.intersection(new_hashes)
    assert len(leakage) == 0, f"CRITICAL LEAKAGE: {len(leakage)} samples overlap with validation!"

    # 7. Write V5 augmented manifest
    v5_all = list(orig_rows)
    for r in new_rows:
        v5_all.append(r.to_csv_dict())

    v5_path = Path("data/manifests/v5_augmented_manifest.csv")
    with open(v5_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(v5_all)

    train_total = sum(1 for r in v5_all if r["split"] == "train")
    val_total = sum(1 for r in v5_all if r["split"] == "val")

    print("\n=================================================================")
    print(" V5 Augmented Manifest Successfully Assembled!")
    print("=================================================================")
    print(f" Manifest Path:        {v5_path}")
    print(f" Total Samples:        {len(v5_all):,}")
    print(f" Training Samples:     {train_total:,} (23,999 original + 5,000 new)")
    print(f" Validation Samples:   {val_total:,} (100% strictly identical original 6,001)")
    print(" Benchmark Overlap:    EXACTLY 0 (Verified against dalle3.csv)")
    print(" Validation Overlap:   EXACTLY 0 (Verified via SHA-256)")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
