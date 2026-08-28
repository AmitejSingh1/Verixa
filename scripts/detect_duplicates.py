from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.dedupe import detect_manifest_duplicates
from verixa.data.manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect exact and near-duplicate images in a manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--near-threshold", type=int, default=5)
    parser.add_argument("--max-near-pairs", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = detect_manifest_duplicates(
        rows=read_manifest(args.manifest),
        near_threshold=args.near_threshold,
        max_near_pairs=args.max_near_pairs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
