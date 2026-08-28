from __future__ import annotations

from verixa.utils.hashing import hamming_distance_hex


def test_hamming_distance_hex() -> None:
    assert hamming_distance_hex("0f", "0f") == 0
    assert hamming_distance_hex("00", "ff") == 8
