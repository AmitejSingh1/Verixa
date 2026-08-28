from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.manifest import manifest_stats, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report counts and leakage indicators for a manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = manifest_stats(read_manifest(args.manifest))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
