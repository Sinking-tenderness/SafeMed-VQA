from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import (
    assign_splits,
    build_index_records,
    ensure_dir,
    load_vqa_rad_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VQA-RAD dataset indices for SafeMed-VQA Pro++.")
    parser.add_argument("--config", type=str, default="configs/datagen.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    seed = int(config.get("seed", 42))
    max_samples = dataset_cfg.get("max_samples")

    records = load_vqa_rad_records(
        dataset_root=dataset_cfg.get("dataset_root"),
        annotations_file=dataset_cfg.get("annotations_file"),
        images_dir=dataset_cfg.get("images_dir"),
    )
    if max_samples:
        records = records[: int(max_samples)]

    split_map = assign_splits(records, val_ratio=float(dataset_cfg.get("val_ratio", 0.0)), seed=seed)
    if args.limit is not None:
        split_map["train"] = split_map["train"][: args.limit]
        split_map["val"] = split_map["val"][: args.limit]
    train_records = build_index_records(split_map["train"])
    val_records = build_index_records(split_map["val"])
    test_records = build_index_records(split_map["test"])

    for key in ["train_index", "val_index", "test_index"]:
        ensure_dir((ROOT / output_cfg[key]).parent)

    write_jsonl(ROOT / output_cfg["train_index"], train_records)
    write_jsonl(ROOT / output_cfg["val_index"], val_records)
    write_jsonl(ROOT / output_cfg["test_index"], test_records)

    print(
        f"Prepared dataset indices: train={len(train_records)} val={len(val_records)} test={len(test_records)}"
    )


if __name__ == "__main__":
    main()
