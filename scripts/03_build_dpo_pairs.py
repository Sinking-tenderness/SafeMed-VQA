from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.dataset_utils import ROOT, ensure_dir, read_jsonl, write_jsonl
from src.image_augment import create_augmented_sample
from src.model_utils import generate_structured_output, load_processor, load_student_model
from src.prompt_templates import build_teacher_messages, make_file_image_url
from src.siliconflow_client import SiliconFlowClient


STRESS_DEGRADATIONS = [
    "random_crop",
    "local_occlusion",
    "center_mask",
    "border_truncate",
    "question_replacement",
    "cross_sample_mismatch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs from high-risk stress samples.")
    parser.add_argument("--config", type=str, default="configs/dpo.yaml")
    parser.add_argument("--datagen-config", type=str, default="configs/datagen.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def prepare_stress_samples(
    stress_source_path: Path,
    fallback_index_path: Path,
    augmented_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    if stress_source_path.exists():
        return read_jsonl(stress_source_path)
    base_records = read_jsonl(fallback_index_path)
    rng = random.Random(seed)
    synthetic: list[dict[str, Any]] = []
    for sample in base_records:
        degradation = rng.choice(STRESS_DEGRADATIONS)
        synthetic.append(
            create_augmented_sample(
                sample=sample,
                pool=base_records,
                output_dir=augmented_root,
                severity="high",
                degradation_type=degradation,
                rng=rng,
            )
        )
    return synthetic


def should_keep_pair(teacher_output: dict[str, Any], student_output: dict[str, Any]) -> bool:
    if teacher_output["decision"] == "abstain" and student_output["decision"] == "answer":
        return True
    if teacher_output["risk_level"] == "high" and student_output["decision"] == "answer":
        return True
    return False


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.datagen_config).open("r", encoding="utf-8") as handle:
        datagen_config = yaml.safe_load(handle)

    data_cfg = config["data"]
    model_cfg = config["model"]
    output_path = ROOT / data_cfg["train_pairs_jsonl"]
    ensure_dir(output_path.parent)

    augmented_root = ROOT / "data" / "stress_test" / "generated_images"
    ensure_dir(augmented_root)
    stress_samples = prepare_stress_samples(
        stress_source_path=ROOT / data_cfg["stress_source_jsonl"],
        fallback_index_path=ROOT / "data" / "interim" / "val_index.jsonl",
        augmented_root=augmented_root,
        seed=int(config.get("seed", 42)),
    )
    if args.limit is not None:
        stress_samples = stress_samples[: args.limit]

    processor = load_processor(model_cfg["model_name"])
    model = load_student_model(
        model_name=model_cfg["model_name"],
        torch_dtype="bfloat16",
        adapter_path=ROOT / model_cfg["sft_checkpoint"],
        trainable_adapter=False,
    )
    teacher = SiliconFlowClient(
        model=datagen_config.get("teacher", {}).get("model", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
        max_retries=int(datagen_config.get("teacher", {}).get("max_retries", 3)),
        error_log_path=ROOT / datagen_config["output"]["datagen_error_log"],
    )

    pairs: list[dict[str, Any]] = []
    for sample in tqdm(stress_samples, desc="Building DPO pairs"):
        teacher_output = teacher.complete_json(
            build_teacher_messages(sample, make_file_image_url(sample["image_path"])),
            temperature=float(datagen_config.get("teacher", {}).get("temperature", 0.2)),
            max_tokens=int(datagen_config.get("teacher", {}).get("max_tokens", 800)),
        )
        student_output, raw_student_text = generate_structured_output(
            model,
            processor,
            sample["image_path"],
            sample["question"],
        )
        if not should_keep_pair(teacher_output, student_output):
            continue
        pairs.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "chosen": json.dumps(teacher_output, ensure_ascii=False),
                "rejected": raw_student_text,
                "teacher_output": teacher_output,
                "student_output": student_output,
            }
        )

    write_jsonl(output_path, pairs)
    print(f"Wrote {len(pairs)} DPO pairs to {output_path}")


if __name__ == "__main__":
    main()
