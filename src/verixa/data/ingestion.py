from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verixa.data.schema import (
    MANIFEST_COLUMNS,
    ManifestRow,
    manifest_path_to_str,
    parse_binary_label,
)
from verixa.utils.hashing import sha256_file
from verixa.utils.images import IMAGE_EXTENSIONS, resize_and_save_jpeg


def ingest_local_image_tree(
    root: Path,
    source_dataset: str,
    label_map: dict[str, Any],
    output_root: Path,
    manifest_path: Path,
    limit_per_class: int | None = None,
    seed: int = 1337,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    parsed_label_map = _parse_optional_label_map(label_map)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in _iter_image_files(root):
        class_name = _class_name_for(root, path)
        if class_name in parsed_label_map and parsed_label_map[class_name] is not None:
            by_class[class_name].append(path)

    selected: list[tuple[str, Path]] = []
    for class_name, paths in sorted(by_class.items()):
        shuffled = list(paths)
        rng.shuffle(shuffled)
        if limit_per_class is not None:
            shuffled = shuffled[:limit_per_class]
        selected.extend((class_name, path) for path in shuffled)

    rng.shuffle(selected)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    counts: Counter[str] = Counter()
    corrupt: list[str] = []
    duplicate_hashes: Counter[str] = Counter()

    for idx, (class_name, source_path) in enumerate(selected):
        label = parsed_label_map[class_name]
        if label is None:
            continue
        destination = output_root / class_name / f"{source_dataset.lower()}_{idx:07d}.jpg"
        try:
            resize_and_save_jpeg(source_path, destination, size=(224, 224), quality=jpeg_quality)
        except Exception:
            corrupt.append(str(source_path))
            continue

        digest = sha256_file(destination)
        duplicate_hashes[digest] += 1
        rows.append(
            ManifestRow(
                image_path=manifest_path_to_str(destination),
                label=label,
                source_dataset=source_dataset,
                split="unassigned",
                generator=None,
                original_id=str(source_path.relative_to(root)),
                sha256=digest,
            )
        )
        counts[f"{class_name}:{label}"] += 1

    _write_manifest(manifest_path, rows)

    duplicates = sum(count - 1 for count in duplicate_hashes.values() if count > 1)
    return {
        "source_dataset": source_dataset,
        "root": str(root),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "selected": len(selected),
        "written": len(rows),
        "corrupt_or_unreadable": len(corrupt),
        "corrupt_examples": corrupt[:25],
        "class_label_counts": dict(sorted(counts.items())),
        "exact_duplicate_images_after_resize": duplicates,
        "disk_usage_mb": _directory_size_mb(output_root),
    }


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def _parse_optional_label_map(label_map: dict[str, Any]) -> dict[str, int | None]:
    parsed: dict[str, int | None] = {}
    for key, value in label_map.items():
        if key.startswith("_"):
            continue
        if value is None or str(value).lower() in {"exclude", "skip", "ignore"}:
            parsed[key] = None
        else:
            parsed[key] = parse_binary_label(value)
    return parsed


def _iter_image_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _class_name_for(root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root)
    if len(relative.parts) <= 1:
        return "__root__"
    return relative.parts[0]


def _directory_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / 1024**2, 3)
