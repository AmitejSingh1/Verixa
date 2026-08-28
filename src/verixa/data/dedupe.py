from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verixa.utils.hashing import hamming_distance_hex, image_average_hash


def detect_manifest_duplicates(
    rows: list[dict[str, str]],
    near_threshold: int = 5,
    max_near_pairs: int = 500,
) -> dict[str, Any]:
    exact_groups = _exact_duplicate_groups(rows)
    hash_rows = _rows_with_average_hash(rows)
    near_pairs = _near_duplicate_pairs(hash_rows, near_threshold, max_near_pairs)
    split_leakage = _split_leakage_for_pairs(rows, exact_groups, near_pairs)

    return {
        "total_rows": len(rows),
        "exact_duplicate_groups": exact_groups[:100],
        "exact_duplicate_group_count": len(exact_groups),
        "near_duplicate_threshold": near_threshold,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_pair_count_reported": len(near_pairs),
        "near_duplicate_pair_limit": max_near_pairs,
        "split_leakage": split_leakage,
    }


def _exact_duplicate_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = row.get("sha256", "")
        if digest:
            grouped[digest].append(row)

    groups: list[dict[str, Any]] = []
    for digest, group in grouped.items():
        if len(group) <= 1:
            continue
        groups.append(
            {
                "sha256": digest,
                "count": len(group),
                "labels": sorted({row.get("label", "") for row in group}),
                "splits": sorted({row.get("split", "") for row in group}),
                "paths": [row.get("image_path", "") for row in group[:10]],
            }
        )
    return sorted(groups, key=lambda item: item["count"], reverse=True)


def _rows_with_average_hash(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    hashed_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = Path(row.get("image_path", ""))
        if not image_path.exists():
            continue
        try:
            average_hash = image_average_hash(image_path)
        except Exception:
            continue
        hashed_row = dict(row)
        hashed_row["average_hash"] = average_hash
        hashed_rows.append(hashed_row)
    return hashed_rows


def _near_duplicate_pairs(
    rows: list[dict[str, str]],
    threshold: int,
    limit: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen_hashes: list[tuple[str, dict[str, str]]] = []

    for row in rows:
        current_hash = row["average_hash"]
        for previous_hash, previous_row in seen_hashes:
            distance = hamming_distance_hex(current_hash, previous_hash)
            if distance <= threshold and row.get("sha256") != previous_row.get("sha256"):
                pairs.append(
                    {
                        "distance": distance,
                        "left": _pair_row(previous_row),
                        "right": _pair_row(row),
                    }
                )
                if len(pairs) >= limit:
                    return pairs
        seen_hashes.append((current_hash, row))

    return pairs


def _split_leakage_for_pairs(
    rows: list[dict[str, str]],
    exact_groups: list[dict[str, Any]],
    near_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_cross_split = [
        group for group in exact_groups if len(set(group.get("splits", [])) - {""}) > 1
    ]
    near_cross_split = [
        pair
        for pair in near_pairs
        if pair["left"].get("split") and pair["right"].get("split")
        and pair["left"].get("split") != pair["right"].get("split")
    ]
    split_counts = Counter(row.get("split", "") for row in rows)
    return {
        "split_counts": dict(sorted(split_counts.items())),
        "exact_cross_split_duplicate_groups": exact_cross_split[:50],
        "near_cross_split_pairs": near_cross_split[:50],
    }


def _pair_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "image_path": row.get("image_path", ""),
        "label": row.get("label", ""),
        "source_dataset": row.get("source_dataset", ""),
        "split": row.get("split", ""),
        "generator": row.get("generator", ""),
        "original_id": row.get("original_id", ""),
        "sha256": row.get("sha256", ""),
        "average_hash": row.get("average_hash", ""),
    }
