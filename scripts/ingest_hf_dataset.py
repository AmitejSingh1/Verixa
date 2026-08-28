from __future__ import annotations

import argparse
import json
from pathlib import Path

from verixa.data.hf_ingestion import ingest_hf_streaming_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a Hugging Face image dataset into resized 224x224 JPEGs."
    )
    parser.add_argument("--dataset", default="saberzl/SID_Set")
    parser.add_argument("--source-dataset", default="SID_Set")
    parser.add_argument("--split", default="train")
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit-per-label", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer-size", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--id-field", default="img_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    stats = ingest_hf_streaming_dataset(
        dataset_name=args.dataset,
        source_dataset=args.source_dataset,
        split=args.split,
        label_map=label_map,
        output_root=args.output_root,
        manifest_path=args.manifest,
        limit_per_label=args.limit_per_label,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        jpeg_quality=args.jpeg_quality,
        image_field=args.image_field,
        label_field=args.label_field,
        id_field=args.id_field,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
