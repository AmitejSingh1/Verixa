from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.manifest import assign_fixed_split, read_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed leakage-checked train/validation split from a manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--stats-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    split_rows, stats = assign_fixed_split(
        rows=rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_manifest(args.out, split_rows)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
