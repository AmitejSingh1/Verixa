from __future__ import annotations

from verixa.data.hf_ingestion import _parse_optional_label_map


def test_parse_optional_label_map_supports_exclude() -> None:
    assert _parse_optional_label_map({"0": 0, "1": 1, "2": "exclude", "_note": "x"}) == {
        "0": 0,
        "1": 1,
        "2": None,
    }


def test_parse_optional_label_map_none_values_excluded_from_sorted_target_counts() -> None:
    """Regression: sorted() must not receive None mixed with int (pre-existing bug)."""
    parsed = _parse_optional_label_map({"0": 0, "1": 1, "2": "exclude"})
    # Simulate what ingest_hf_streaming_dataset does to build target_counts
    target_counts = {
        label: 10
        for label in sorted(v for v in set(parsed.values()) if v is not None)
    }
    assert target_counts == {0: 10, 1: 10}
