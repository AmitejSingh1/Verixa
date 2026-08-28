from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.ingestion import ingest_local_image_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize a local image-folder dataset to 224x224 JPEGs and emit a manifest."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument(
        "--label-map",
        type=Path,
        required=True,
        help="JSON object mapping class folder names to 0=real or 1=ai_generated.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    stats = ingest_local_image_tree(
        root=args.root,
        source_dataset=args.source_dataset,
        label_map=label_map,
        output_root=args.output_root,
        manifest_path=args.manifest,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
