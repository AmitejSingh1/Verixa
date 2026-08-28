"""Unit tests for WildFake ingestion sampling logic (pure logic, no network)."""
from __future__ import annotations

import csv
from pathlib import Path

from verixa.data.wildfake_ingestion import sample_wildfake_rows


def _write_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "train_metadata.csv"
    fieldnames = [
        "Generator",
        "Architecture",
        "Weight",
        "Category",
        "IsAdvanced",
        "IsFake",
        "Image_path",
        "Num",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(arch: str, is_fake: str, idx: int) -> dict[str, str]:
    return {
        "Generator": "Diffusion_based" if arch != "BigGAN" else "GAN_based",
        "Architecture": arch,
        "Weight": arch,
        "Category": arch,
        "IsAdvanced": "0",
        "IsFake": is_fake,
        "Image_path": f"./{arch}/{idx}.jpg",
        "Num": str(idx),
    }


def test_sample_wildfake_rows_respects_caps(tmp_path: Path) -> None:
    rows = (
        [_row("DDPM", "1", i) for i in range(100)]
        + [_row("BigGAN", "1", i) for i in range(50)]
        + [_row("Real", "0", i) for i in range(80)]
    )
    csv_path = _write_csv(tmp_path, rows)

    selected = sample_wildfake_rows(
        metadata_csv=csv_path,
        caps_per_architecture={"DDPM": 10, "BigGAN": 5},
        real_cap=7,
        seed=42,
    )
    assert len(selected) == 22
    fake_counts = sum(1 for r in selected if r["IsFake"] == "1")
    real_counts = sum(1 for r in selected if r["IsFake"] == "0")
    assert fake_counts == 15
    assert real_counts == 7


def test_sample_wildfake_rows_underfilled_pool(tmp_path: Path) -> None:
    rows = [_row("DDPM", "1", i) for i in range(3)] + [_row("Real", "0", i) for i in range(2)]
    csv_path = _write_csv(tmp_path, rows)

    selected = sample_wildfake_rows(
        metadata_csv=csv_path,
        caps_per_architecture={"DDPM": 999},
        real_cap=999,
        seed=0,
    )
    assert len(selected) == 5
