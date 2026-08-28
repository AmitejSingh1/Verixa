from __future__ import annotations

from verixa.data.manifest import assign_fixed_split, manifest_stats


def _row(index: int, label: int, source: str = "unit", sha: str | None = None) -> dict[str, str]:
    return {
        "image_path": f"image_{index}.jpg",
        "label": str(label),
        "source_dataset": source,
        "split": "unassigned",
        "generator": "",
        "original_id": str(index),
        "sha256": sha or f"sha_{index}",
    }


def test_assign_fixed_split_keeps_duplicate_hashes_together() -> None:
    rows = [_row(0, 0, sha="same"), _row(1, 0, sha="same")]
    rows.extend(_row(index, index % 2) for index in range(2, 20))

    split_rows, stats = assign_fixed_split(rows, val_fraction=0.2, seed=7)
    splits_for_duplicate = {row["split"] for row in split_rows if row["sha256"] == "same"}

    assert len(splits_for_duplicate) == 1
    assert stats["split_leakage_duplicate_hashes"] == []


def test_manifest_stats_counts_labels_sources_and_duplicates() -> None:
    rows = [_row(0, 0, source="a", sha="dup"), _row(1, 1, source="b", sha="dup")]

    stats = manifest_stats(rows)

    assert stats["total_images"] == 2
    assert stats["label_counts"] == {"0": 1, "1": 1}
    assert stats["source_counts"] == {"a": 1, "b": 1}
    assert stats["exact_duplicate_images"] == 1
