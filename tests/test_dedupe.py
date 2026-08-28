from __future__ import annotations

from pathlib import Path

from PIL import Image

from verixa.data.dedupe import detect_manifest_duplicates
from verixa.utils.hashing import sha256_file


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG")


def _row(path: Path, split: str, digest: str) -> dict[str, str]:
    return {
        "image_path": str(path),
        "label": "0",
        "source_dataset": "unit",
        "split": split,
        "generator": "",
        "original_id": path.name,
        "sha256": digest,
    }


def test_detect_manifest_duplicates_reports_exact_split_leakage(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    _make_image(image, (255, 0, 0))
    digest = sha256_file(image)
    rows = [_row(image, "train", digest), _row(image, "val", digest)]

    report = detect_manifest_duplicates(rows)

    assert report["exact_duplicate_group_count"] == 1
    assert report["split_leakage"]["exact_cross_split_duplicate_groups"]
