from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAKE via KaggleHub cache.")
    parser.add_argument(
        "--dataset",
        default="birdy654/cifake-real-and-ai-generated-synthetic-images",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/cifake_download.json"))
    return parser.parse_args()


def main() -> int:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install kagglehub before downloading CIFAKE.") from exc

    args = parse_args()
    path = Path(kagglehub.dataset_download(args.dataset))
    report = {
        "dataset": args.dataset,
        "local_cache_path": str(path),
        "note": "CIFAKE is cached outside the repository. Do not copy raw originals into data/.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
