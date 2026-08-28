from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verixa.data.schema import MANIFEST_COLUMNS, parse_binary_label


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is empty: {path}")
        missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
        return [{column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in reader]


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def merge_manifests(paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_manifest(path))

    return rows, manifest_stats(rows)


def assign_fixed_split(
    rows: list[dict[str, str]],
    val_fraction: float = 0.2,
    seed: int = 1337,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    _validate_labels(rows)
    groups = _group_by_sha256(rows)
    _raise_on_label_conflicts(groups)

    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups.values():
        first = group[0]
        key = (
            first["label"],
            first["source_dataset"],
            first.get("generator", "") or "",
        )
        strata[key].append(group)

    output_rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    stratum_counts: dict[str, dict[str, int]] = {}

    for key, stratum_groups in sorted(strata.items()):
        shuffled = sorted(stratum_groups, key=lambda g: g[0]["sha256"])
        rng.shuffle(shuffled)
        val_count = round(len(shuffled) * val_fraction)
        if len(shuffled) > 1:
            val_count = max(1, val_count)
        else:
            val_count = 0

        val_ids = {id(group) for group in shuffled[:val_count]}
        stratum_name = "|".join(key)
        stratum_counts[stratum_name] = {"train": 0, "val": 0}

        for group in shuffled:
            split = "val" if id(group) in val_ids else "train"
            for row in group:
                split_row = dict(row)
                split_row["split"] = split
                output_rows.append(split_row)
                split_counts[split] += 1
                stratum_counts[stratum_name][split] += 1

    stats = manifest_stats(output_rows)
    stats.update(
        {
            "seed": seed,
            "val_fraction": val_fraction,
            "split_counts": dict(sorted(split_counts.items())),
            "stratum_split_counts": stratum_counts,
            "exact_duplicate_groups": sum(1 for group in groups.values() if len(group) > 1),
        }
    )
    return output_rows, stats


def manifest_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    generators: Counter[str] = Counter()
    sha_counts: Counter[str] = Counter()

    for row in rows:
        labels[str(row.get("label", ""))] += 1
        sources[row.get("source_dataset", "")] += 1
        splits[row.get("split", "")] += 1
        generator = row.get("generator", "") or "__missing__"
        generators[generator] += 1
        digest = row.get("sha256", "")
        if digest:
            sha_counts[digest] += 1

    duplicate_images = sum(count - 1 for count in sha_counts.values() if count > 1)
    split_leakage = _find_split_leakage(rows)

    return {
        "total_images": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "source_counts": dict(sorted(sources.items())),
        "split_counts": dict(sorted(splits.items())),
        "generator_counts_top50": dict(generators.most_common(50)),
        "exact_duplicate_images": duplicate_images,
        "split_leakage_duplicate_hashes": split_leakage,
    }


def _validate_labels(rows: list[dict[str, str]]) -> None:
    for row in rows:
        parse_binary_label(row.get("label", ""))


def _group_by_sha256(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        digest = row.get("sha256") or f"__row_{index}"
        groups[digest].append(row)
    return groups


def _raise_on_label_conflicts(groups: dict[str, list[dict[str, str]]]) -> None:
    conflicts: list[str] = []
    for digest, group in groups.items():
        labels = {row.get("label", "") for row in group}
        if len(labels) > 1:
            conflicts.append(digest)
    if conflicts:
        raise ValueError(
            "Exact duplicate image hashes have conflicting labels. "
            f"Resolve before splitting. Example hashes: {conflicts[:10]}"
        )


def _find_split_leakage(rows: list[dict[str, str]]) -> list[str]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        digest = row.get("sha256", "")
        split = row.get("split", "")
        if digest and split:
            by_hash[digest].add(split)
    return sorted(digest for digest, splits in by_hash.items() if len(splits) > 1)[:50]
