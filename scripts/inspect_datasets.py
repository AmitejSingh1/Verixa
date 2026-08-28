from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.inspection import inspect_hf_dataset, inspect_modelscope_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect remote dataset metadata before sampling or ingestion."
    )
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_inspection.json"))
    parser.add_argument("--sid-name", default="saberzl/SID_Set")
    parser.add_argument("--wildfake-name", default="hy2628982280/WildFake")
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument(
        "--include-wildfake",
        action="store_true",
        help="Attempt ModelScope WildFake inspection after translation/access is ready.",
    )
    parser.add_argument(
        "--skip-wildfake",
        action="store_true",
        help="Backward-compatible alias for leaving WildFake out of this pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {
        "sid_set": inspect_hf_dataset(args.sid_name, sample_limit=args.sample_limit),
        "wildfake": None,
        "manual_checkpoints": [
            "Confirm SID_Set category meanings before binary REAL vs AI-GENERATED mapping.",
            "Use ModelScope's translation flow for WildFake before sampling.",
            "Do not touch the held-out benchmark until Phase 5.",
        ],
    }

    if args.include_wildfake and not args.skip_wildfake:
        report["wildfake"] = inspect_modelscope_dataset(
            args.wildfake_name, sample_limit=args.sample_limit
        )
    else:
        report["wildfake"] = {
            "dataset": args.wildfake_name,
            "backend": "modelscope",
            "skipped": True,
            "reason": "WildFake inspection is opt-in until ModelScope translation/access is ready.",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote dataset inspection report to {args.out}")

    sid_warning = report["sid_set"].get("warnings", [])
    if sid_warning:
        print("SID_Set warnings:")
        for warning in sid_warning:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
