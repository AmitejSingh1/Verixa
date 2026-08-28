from __future__ import annotations

from verixa.data.hf_ingestion import _parse_optional_label_map


def test_parse_optional_label_map_supports_exclude() -> None:
    assert _parse_optional_label_map({"0": 0, "1": 1, "2": "exclude", "_note": "x"}) == {
        "0": 0,
        "1": 1,
        "2": None,
    }
