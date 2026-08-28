"""CLI script: ingest a controlled WildFake subset into the pilot manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a controlled WildFake subset from ModelScope "
            "using streaming metadata rows + per-file image download."
        )
    )
    parser.add_argument(
        "--dataset",
        default="hy2628982280/WildFake",
        help="ModelScope dataset in namespace/name form.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-cap", type=int, default=1000)
    parser.add_argument(
        "--arch-caps",
        type=str,
        default="",
        help="JSON dict string or path to JSON file containing architecture caps.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from verixa.data.wildfake_ingestion import ingest_wildfake

    if not args.arch_caps:
        caps_per_architecture = {
            "DDPM": 400,
            "BigGAN": 300,
            "ADM": 200,
            "DALLE": 100,
            "Midjourney": 100,
        }
    elif Path(args.arch_caps).is_file():
        caps_per_architecture = json.loads(Path(args.arch_caps).read_text(encoding="utf-8"))
    else:
        caps_per_architecture = json.loads(args.arch_caps)

    stats = ingest_wildfake(
        dataset_name=args.dataset,
        output_root=args.output_root,
        manifest_path=args.manifest,
        caps_per_architecture=caps_per_architecture,
        real_cap=args.real_cap,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        download_retries=args.download_retries,
        download_timeout=args.download_timeout,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
