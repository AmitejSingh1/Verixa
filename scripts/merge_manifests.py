from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.manifest import merge_manifests, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge source manifests into one combined manifest."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, stats = merge_manifests(args.manifest)
    write_manifest(args.out, rows)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
