from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

MANIFEST_COLUMNS = [
    "image_path",
    "label",
    "source_dataset",
    "split",
    "generator",
    "original_id",
    "sha256",
]


class BinaryLabel(IntEnum):
    REAL = 0
    AI_GENERATED = 1


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    label: int
    source_dataset: str
    split: str
    generator: str | None
    original_id: str
    sha256: str

    def to_csv_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["generator"] = values["generator"] or ""
        return values


def parse_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean labels are not accepted for binary image labels.")

    if isinstance(value, int) and value in {0, 1}:
        return value

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"0", "real", "authentic", "human", "non_ai", "non_aigc"}:
            return BinaryLabel.REAL.value
        if normalized in {"1", "fake", "ai", "aigc", "synthetic", "ai_generated"}:
            return BinaryLabel.AI_GENERATED.value

    raise ValueError(f"Cannot map label {value!r} to 0=real or 1=ai_generated.")


def manifest_path_to_str(path: Path) -> str:
    return str(path.as_posix())
