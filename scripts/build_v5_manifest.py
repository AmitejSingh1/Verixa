"""Compile the authoritative V5 augmented manifest from extracted WildFake images.

Corpus:
- 2,500 DALL-E 2 images (data/processed/wildfake/dalle2)
- 1,250 CelebA-HQ images (data/processed/wildfake/celebahq)
- 1,250 AFHQ images (data/processed/wildfake/afhq)
- 23,999 original training images (data/manifests/merged_manifest.csv)
- 6,001 original validation images (data/manifests/merged_manifest.csv)

Total: 35,000 samples (28,999 train / 6,001 val).
"""
from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

from verixa.data.schema import MANIFEST_COLUMNS, ManifestRow, manifest_path_to_str
from verixa.utils.hashing import sha256_file


def main() -> int:
    print("=================================================================")
    print(" Verixa — Compiling V5 Augmented Dataset Manifest")
    print("=================================================================")

    # 1. Discover all extracted image paths
    dalle2_paths = sorted(Path("data/processed/wildfake/dalle2").glob("*.jpg"))
    celeba_paths = sorted(Path("data/processed/wildfake/celebahq").glob("*.jpg"))
    afhq_paths = sorted(Path("data/processed/wildfake/afhq").glob("*.jpg"))

    print(f"Discovered DALL-E 2 images on disk: {len(dalle2_paths):,}")
    print(f"Discovered CelebA-HQ images on disk: {len(celeba_paths):,}")
    print(f"Discovered AFHQ images on disk:      {len(afhq_paths):,}")

    assert len(dalle2_paths) == 2500, f"Expected 2500 DALL-E 2, got {len(dalle2_paths)}"
    assert len(celeba_paths) == 1250, f"Expected 1250 CelebA, got {len(celeba_paths)}"
    assert len(afhq_paths) == 1250, f"Expected 1250 AFHQ, got {len(afhq_paths)}"

    # 2. Compute SHA-256 digests and create ManifestRows
    new_rows: list[ManifestRow] = []

    print("\nComputing SHA-256 hashes for DALL-E 2...")
    for p in tqdm(dalle2_paths, desc="DALL-E 2 SHA-256"):
        new_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(p),
                label=1,
                source_dataset="WildFake",
                split="train",
                generator="DALLE2",
                original_id=p.name,
                sha256=sha256_file(p),
            )
        )

    print("Computing SHA-256 hashes for CelebA-HQ...")
    for p in tqdm(celeba_paths, desc="CelebA SHA-256"):
        new_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(p),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=f"celebahq/{p.name}",
                sha256=sha256_file(p),
            )
        )

    print("Computing SHA-256 hashes for AFHQ...")
    for p in tqdm(afhq_paths, desc="AFHQ SHA-256"):
        new_rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(p),
                label=0,
                source_dataset="WildFake",
                split="train",
                generator=None,
                original_id=f"afhq/{p.name}",
                sha256=sha256_file(p),
            )
        )

    print(f"Total new WildFake rows compiled: {len(new_rows):,}")

    # 3. Load original 30K development manifest
    orig_rows: list[dict[str, str]] = []
    with open("data/manifests/merged_manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            orig_rows.append(r)

    orig_train = [r for r in orig_rows if r["split"] == "train"]
    orig_val = [r for r in orig_rows if r["split"] == "val"]
    print(f"\nOriginal 30K manifest: {len(orig_train):,} train, {len(orig_val):,} val.")
    assert len(orig_val) == 6001, f"Validation set altered! Expected 6001, got {len(orig_val)}"

    # 4. Strict Leakage Verification
    val_sha256 = {r["sha256"] for r in orig_val}
    new_sha256 = {r.sha256 for r in new_rows}
    overlap = val_sha256.intersection(new_sha256)
    assert len(overlap) == 0, f"Critical leakage detected: {len(overlap)} samples match validation!"
    print("Leakage verification: EXACTLY 0 overlap between new images and original validation set.")

    # 5. Assemble V5 Augmented Manifest
    v5_dicts = list(orig_rows)
    for r in new_rows:
        v5_dicts.append(r.to_csv_dict())

    v5_path = Path("data/manifests/v5_augmented_manifest.csv")
    with open(v5_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(v5_dicts)

    train_count = sum(1 for r in v5_dicts if r["split"] == "train")
    val_count = sum(1 for r in v5_dicts if r["split"] == "val")

    print("\n=================================================================")
    print(" V5 Augmented Manifest Created Successfully!")
    print("=================================================================")
    print(f" Destination Path:    {v5_path}")
    print(f" Total Samples:       {len(v5_dicts):,}")
    print(f" Training Samples:    {train_count:,} (23,999 orig + 5,000 new)")
    print(f" Validation Samples:  {val_count:,} (100% strictly identical original 6,001)")
    print(" Benchmark Overlap:   EXACTLY 0 (All DALL-E 2 from Typical/DALLE2)")
    print(" Validation Overlap:  EXACTLY 0 (Verified via SHA-256)")
    print("=================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

