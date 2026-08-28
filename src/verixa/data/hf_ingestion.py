from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from verixa.data.schema import (
    MANIFEST_COLUMNS,
    ManifestRow,
    manifest_path_to_str,
    parse_binary_label,
)
from verixa.utils.hashing import sha256_file
from verixa.utils.images import save_pil_jpeg

EXCLUDE_VALUES = {"exclude", "skip", "ignore", None}


def ingest_hf_streaming_dataset(
    dataset_name: str,
    source_dataset: str,
    split: str,
    label_map: dict[str, Any],
    output_root: Path,
    manifest_path: Path,
    limit_per_label: int,
    seed: int = 1337,
    shuffle_buffer_size: int = 2000,
    jpeg_quality: int = 90,
    image_field: str = "image",
    label_field: str = "label",
    id_field: str = "img_id",
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before streaming Hugging Face datasets.") from exc

    parsed_label_map = _parse_optional_label_map(label_map)
    target_counts = {
        label: limit_per_label
        for label in sorted(v for v in set(parsed_label_map.values()) if v is not None)
    }
    counts: Counter[int] = Counter()
    skipped_source_labels: Counter[str] = Counter()
    corrupt = 0
    processed_rows = 0
    written_rows = 0
    rows: list[ManifestRow] = []

    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(dataset_name, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    progress_total = sum(target_counts.values())
    progress = tqdm(total=progress_total, desc=f"ingest {source_dataset}:{split}")

    try:
        for row_index, row in enumerate(stream):
            processed_rows += 1
            raw_label = str(row[label_field])
            mapped_label = parsed_label_map.get(raw_label)
            if mapped_label is None:
                skipped_source_labels[raw_label] += 1
                continue
            if counts[mapped_label] >= target_counts[mapped_label]:
                continue

            image = row[image_field]
            original_id = str(row.get(id_field, row_index))
            destination = (
                output_root
                / f"label_{mapped_label}"
                / f"{source_dataset.lower()}_{split}_{written_rows:07d}.jpg"
            )
            try:
                save_pil_jpeg(image, destination, size=(224, 224), quality=jpeg_quality)
            except Exception:
                corrupt += 1
                continue

            digest = sha256_file(destination)
            rows.append(
                ManifestRow(
                    image_path=manifest_path_to_str(destination),
                    label=mapped_label,
                    source_dataset=source_dataset,
                    split="unassigned",
                    generator=None,
                    original_id=original_id,
                    sha256=digest,
                )
            )
            counts[mapped_label] += 1
            written_rows += 1
            progress.update(1)

            if all(counts[label] >= target for label, target in target_counts.items()):
                break
    finally:
        progress.close()

    _write_manifest(manifest_path, rows)
    return {
        "dataset": dataset_name,
        "source_dataset": source_dataset,
        "split": split,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "limit_per_label": limit_per_label,
        "processed_stream_rows": processed_rows,
        "written_rows": written_rows,
        "label_counts": {str(key): counts[key] for key in sorted(counts)},
        "skipped_source_labels": dict(sorted(skipped_source_labels.items())),
        "corrupt_or_unreadable": corrupt,
        "seed": seed,
        "shuffle_buffer_size": shuffle_buffer_size,
        "jpeg_quality": jpeg_quality,
        "disk_usage_mb": _directory_size_mb(output_root),
    }


def _parse_optional_label_map(label_map: dict[str, Any]) -> dict[str, int | None]:
    parsed: dict[str, int | None] = {}
    for key, value in label_map.items():
        if key.startswith("_"):
            continue
        if value in EXCLUDE_VALUES or str(value).lower() in EXCLUDE_VALUES:
            parsed[str(key)] = None
        else:
            parsed[str(key)] = parse_binary_label(value)
    return parsed


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
