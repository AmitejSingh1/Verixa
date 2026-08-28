from __future__ import annotations

import pytest

from verixa.data.schema import BinaryLabel, parse_binary_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, BinaryLabel.REAL),
        (1, BinaryLabel.AI_GENERATED),
        ("real", BinaryLabel.REAL),
        ("authentic", BinaryLabel.REAL),
        ("AI-generated", BinaryLabel.AI_GENERATED),
        ("synthetic", BinaryLabel.AI_GENERATED),
    ],
)
def test_parse_binary_label(raw: object, expected: BinaryLabel) -> None:
    assert parse_binary_label(raw) == expected.value


def test_parse_binary_label_rejects_ambiguous_value() -> None:
    with pytest.raises(ValueError):
        parse_binary_label("tampered")
