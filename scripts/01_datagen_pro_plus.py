from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.dataset_utils import ROOT, ensure_dir, read_jsonl, write_jsonl
from src.image_augment import create_augmented_sample, severity_choices
from src.prompt_templates import build_teacher_messages, make_file_image_url
from src.siliconflow_client import SiliconFlowClient


DEGRADATION_TYPES = [
    "gaussian_blur",
    "speckle_noise",
    "resolution_drop",
    "contrast_brightness_shift",
    "random_crop",
    "local_occlusion",
    "center_mask",
    "border_truncate",
]
MISMATCH_TYPES = ["question_replacement", "image_replacement", "cross_sample_mismatch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SafeMed-VQA Pro++ training data with SiliconFlow teacher.")
    parser.add_argument("--config", type=str, default="configs/datagen.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_sample_variants(
    sample: dict[str, Any],
    pool: list[dict[str, Any]],
    augmented_root: Path,
    cfg: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    variants = [{**sample, "is_counterfactual": False, "degradation_type": "none", "severity": "none"}]
    aug_cfg = cfg.get("augmentation", {})
    weights = aug_cfg.get("severity_weights", {})
    severity_pool = severity_choices(weights)
    mismatch_probability = float(aug_cfg.get("mismatch_probability", 0.25))
    per_sample_variants = int(aug_cfg.get("per_sample_variants", 4))

    for _ in range(per_sample_variants):
        severity = rng.choice(severity_pool)
        if rng.random() < mismatch_probability:
            degradation_type = rng.choice(MISMATCH_TYPES)
        else:
            degradation_type = rng.choice(DEGRADATION_TYPES)
        variants.append(
            create_augmented_sample(
                sample=sample,
                pool=pool,
                output_dir=augmented_root,
                severity=severity,
                degradation_type=degradation_type,
                rng=rng,
            )
        )
    return variants


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_cfg = config["output"]
    train_index_path = ROOT / output_cfg["train_index"]
    augmented_root = ROOT / output_cfg["augmented_root"]
    ensure_dir(augmented_root)

    samples = read_jsonl(train_index_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    rng = random.Random(int(config.get("seed", 42)))
    client = SiliconFlowClient(
        model=config.get("teacher", {}).get("model", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
        max_retries=int(config.get("teacher", {}).get("max_retries", 3)),
        error_log_path=ROOT / output_cfg["datagen_error_log"],
    )

    records_out: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Generating teacher supervision"):
        variants = build_sample_variants(sample, samples, augmented_root, config, rng)
        for variant in variants:
            image_url = make_file_image_url(variant["image_path"])
            messages = build_teacher_messages(variant, image_url)
            teacher_output = client.complete_json(
                messages,
                temperature=float(config.get("teacher", {}).get("temperature", 0.2)),
                max_tokens=int(config.get("teacher", {}).get("max_tokens", 800)),
            )
            record = {
                "sample_id": variant["sample_id"],
                "image_path": variant["image_path"],
                "question": variant["question"],
                "reference_answer": variant.get("answer", ""),
                "split": variant.get("split", "train"),
                "is_counterfactual": bool(variant.get("is_counterfactual")),
                "degradation_type": variant.get("degradation_type", "none"),
                "severity": variant.get("severity", "none"),
                "source_sample_id": variant.get("source_sample_id", variant["sample_id"]),
                "teacher_output": teacher_output,
            }
            records_out.append(record)

    write_jsonl(ROOT / output_cfg["teacher_train_jsonl"], records_out)
    print(f"Wrote {len(records_out)} teacher-supervised records to {ROOT / output_cfg['teacher_train_jsonl']}")


if __name__ == "__main__":
    main()
