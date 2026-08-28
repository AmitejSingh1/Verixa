from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verixa.utils.images import IMAGE_EXTENSIONS, probe_image

AMBIGUOUS_LABEL_TERMS = {
    "tamper",
    "tampered",
    "manipulated",
    "manipulation",
    "edited",
    "edit",
    "splice",
    "spliced",
    "forged",
    "forgery",
}


def inspect_hf_dataset(dataset_name: str, sample_limit: int = 128) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": "huggingface",
        "features": {},
        "splits": {},
        "sample_summary": {},
        "warnings": [],
    }

    try:
        from datasets import ClassLabel, Image, load_dataset, load_dataset_builder
    except ImportError as exc:
        report["error"] = f"Install datasets first: {exc}"
        return report

    try:
        builder = load_dataset_builder(dataset_name)
        info = builder.info
    except Exception as exc:
        report["error"] = f"Could not load Hugging Face dataset builder: {exc}"
        return report

    features = getattr(info, "features", None) or {}
    for name, feature in features.items():
        feature_info: dict[str, Any] = {"type": type(feature).__name__}
        if isinstance(feature, ClassLabel):
            feature_info["names"] = list(feature.names)
            ambiguous = _find_ambiguous_labels(feature.names)
            if ambiguous:
                report["warnings"].append(
                    f"Feature '{name}' has ambiguous label names requiring review: {ambiguous}"
                )
        if isinstance(feature, Image):
            feature_info["decode"] = getattr(feature, "decode", None)
        report["features"][name] = feature_info

    if info.splits:
        for split_name, split_info in info.splits.items():
            report["splits"][split_name] = {
                "num_examples": getattr(split_info, "num_examples", None),
                "num_bytes": getattr(split_info, "num_bytes", None),
            }

    split_names = list(report["splits"]) or ["train"]
    for split_name in split_names[:3]:
        try:
            stream = load_dataset(dataset_name, split=split_name, streaming=True)
            report["sample_summary"][split_name] = _summarize_iterable_rows(
                stream, sample_limit=sample_limit
            )
        except Exception as exc:
            report["sample_summary"][split_name] = {"error": str(exc)}

    if dataset_name.lower().endswith("sid_set") and not report["warnings"]:
        report["warnings"].append(
            "SID_Set still requires manual category-definition review before binary mapping."
        )

    return report


def inspect_modelscope_dataset(dataset_name: str, sample_limit: int = 128) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "backend": "modelscope",
        "sample_limit": sample_limit,
        "warnings": [
            "Use ModelScope's translation flow for WildFake before relying on metadata fields.",
            "Sample across available generator/source metadata; do not bulk-sample one generator.",
        ],
    }

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        report["error"] = f"Install modelscope first: {exc}"
        return report

    try:
        dataset = MsDataset.load(dataset_name)
    except Exception as exc:
        report["error"] = f"Could not load ModelScope dataset metadata: {exc}"
        return report

    report["type"] = type(dataset).__name__
    if isinstance(dataset, dict):
        report["splits"] = list(dataset)
        for split_name, split_data in dataset.items():
            report.setdefault("sample_summary", {})[split_name] = _summarize_iterable_rows(
                split_data, sample_limit=sample_limit
            )
    else:
        report["sample_summary"] = _summarize_iterable_rows(dataset, sample_limit=sample_limit)

    return report


def inspect_local_image_tree(root: Path, sample_limit: int = 512) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {
        "dataset": str(root),
        "backend": "local_image_tree",
        "exists": root.exists(),
        "class_counts": {},
        "formats": {},
        "image_size_samples": [],
        "corrupt_or_unreadable": [],
        "total_images": 0,
    }
    if not root.exists():
        report["error"] = "Root does not exist."
        return report

    class_counts: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    sampled = 0
    corrupt: list[str] = []
    sizes: list[dict[str, Any]] = []

    for image_path in sorted(_iter_image_files(root)):
        class_name = _class_name_for(root, image_path)
        class_counts[class_name] += 1
        if sampled < sample_limit:
            probe = probe_image(image_path)
            if probe["valid"]:
                formats[str(probe["format"])] += 1
                sizes.append(
                    {
                        "path": str(image_path.relative_to(root)),
                        "width": probe["width"],
                        "height": probe["height"],
                        "format": probe["format"],
                    }
                )
            else:
                corrupt.append(str(image_path.relative_to(root)))
            sampled += 1

    report["class_counts"] = dict(sorted(class_counts.items()))
    report["formats"] = dict(sorted(formats.items()))
    report["image_size_samples"] = sizes
    report["corrupt_or_unreadable"] = corrupt
    report["total_images"] = sum(class_counts.values())
    report["warnings"] = _local_dataset_warnings(report["class_counts"])
    return report


def _summarize_iterable_rows(rows: Any, sample_limit: int) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    id_prefix_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    generator_counts: Counter[str] = Counter()
    image_sizes: list[dict[str, Any]] = []
    row_count = 0

    for row in rows:
        row_count += 1
        if isinstance(row, dict):
            key_counts.update(row.keys())
            sampled_label = _first_matching_value(row, ("label", "class"))
            sampled_id = _first_matching_value(row, ("img_id", "image_id", "id", "filename"))
            if sampled_label is not None and sampled_id is not None:
                id_prefix_by_label[str(sampled_label)][_id_prefix(str(sampled_id))] += 1

            for key, value in row.items():
                lower_key = key.lower()
                if "label" in lower_key or "class" in lower_key:
                    label_counts[key][str(value)] += 1
                if "generator" in lower_key or "source" in lower_key or "model" in lower_key:
                    if value is not None:
                        generator_counts[str(value)] += 1
                if hasattr(value, "size") and len(getattr(value, "size", ())) == 2:
                    width, height = value.size
                    image_sizes.append({"field": key, "width": width, "height": height})
        if row_count >= sample_limit:
            break

    warnings: list[str] = []
    for field, counts in label_counts.items():
        ambiguous = _find_ambiguous_labels(counts)
        if ambiguous:
            warnings.append(f"Field '{field}' contains ambiguous sampled labels: {ambiguous}")

    return {
        "sampled_rows": row_count,
        "keys": dict(sorted(key_counts.items())),
        "label_value_counts": {key: dict(values) for key, values in label_counts.items()},
        "id_prefix_by_label": {
            key: dict(values.most_common(25)) for key, values in sorted(id_prefix_by_label.items())
        },
        "generator_or_source_counts": dict(generator_counts.most_common(50)),
        "image_size_samples": image_sizes[:50],
        "warnings": warnings,
    }


def _iter_image_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _class_name_for(root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root)
    if len(relative.parts) <= 1:
        return "__root__"
    return relative.parts[0]


def _find_ambiguous_labels(labels: Any) -> list[str]:
    ambiguous: list[str] = []
    for label in labels:
        normalized = str(label).strip().lower()
        if any(term in normalized for term in AMBIGUOUS_LABEL_TERMS):
            ambiguous.append(str(label))
    return ambiguous


def _first_matching_value(row: dict[str, Any], terms: tuple[str, ...]) -> Any:
    for key, value in row.items():
        lower_key = key.lower()
        if any(term == lower_key or term in lower_key for term in terms):
            return value
    return None


def _id_prefix(identifier: str) -> str:
    if identifier.startswith("full_synthetic"):
        return "full_synthetic"
    if identifier.startswith("tampered"):
        return "tampered"
    if "_" in identifier:
        return identifier.split("_", maxsplit=1)[0]
    return "__no_prefix__"


def _local_dataset_warnings(class_counts: dict[str, int]) -> list[str]:
    warnings: list[str] = []
    ambiguous = _find_ambiguous_labels(class_counts)
    if ambiguous:
        warnings.append(f"Ambiguous class folders require manual label mapping: {ambiguous}")
    if not class_counts:
        warnings.append("No image files were found under the dataset root.")
    return warnings
