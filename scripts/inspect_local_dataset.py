from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.inspection import inspect_local_image_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local image-folder dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/local_dataset_inspection.json"))
    parser.add_argument("--sample-limit", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_local_image_tree(args.root, sample_limit=args.sample_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote local dataset inspection report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
