from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, read_jsonl, write_jsonl
from src.image_augment import (
    SUPPORTED_DEGRADATION_TYPES,
    apply_degradation_with_params,
    load_image,
    normalize_degradation_type,
    normalize_severity,
)


DEFAULT_CONFIG = ROOT / "configs" / "datagen.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build degraded test images/index from clean test_index.jsonl.")
    parser.add_argument("--input-jsonl", type=str, default="data/interim/test_index.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="data/interim/test_degraded_index.jsonl")
    parser.add_argument("--output-image-dir", type=str, default="data/processed/degraded_test_images")
    parser.add_argument("--degradation-types", nargs="+", default=None)
    parser.add_argument("--severities", nargs="+", default=["mild", "medium", "severe"])
    parser.add_argument("--num-degraded-per-sample", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-clean-pairs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_default_degradation_types() -> list[str]:
    if DEFAULT_CONFIG.exists():
        with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        configured = config.get("augmentation", {}).get("degradation_types")
        if configured:
            return [normalize_degradation_type(str(item)) for item in configured]
    return list(SUPPORTED_DEGRADATION_TYPES)


def stable_seed(base_seed: int, sample_id: str, variant_index: int) -> int:
    digest = hashlib.md5(f"{base_seed}:{sample_id}:{variant_index}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def compact_original_record(record: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "sample_id",
        "image_path",
        "question",
        "answer",
        "reference_answer",
        "split",
        "answer_type",
        "phrase_type",
        "source_dataset",
    ]
    return {key: record[key] for key in keep_keys if key in record}


def build_clean_pair_record(record: dict[str, Any]) -> dict[str, Any]:
    clean_path = record.get("image_path")
    result = dict(record)
    result["clean_sample_id"] = record["sample_id"]
    result["clean_image_path"] = clean_path
    result["reference_answer"] = record.get("reference_answer", record.get("answer", ""))
    result["split"] = "test"
    result["is_counterfactual"] = False
    result["degradation_type"] = "none"
    result["severity"] = "none"
    result["degradation_params"] = {}
    return result


def build_degraded_record(
    *,
    record: dict[str, Any],
    output_image_path: Path,
    degradation_type: str,
    severity: str,
    variant_index: int,
    degradation_params: dict[str, Any],
) -> dict[str, Any]:
    clean_sample_id = str(record["sample_id"])
    result: dict[str, Any] = {
        "sample_id": f"{clean_sample_id}__deg__{degradation_type}__{severity}__{variant_index}",
        "clean_sample_id": clean_sample_id,
        "image_path": str(output_image_path.resolve()),
        "clean_image_path": record["image_path"],
        "question": record["question"],
        "reference_answer": record.get("reference_answer", record.get("answer", "")),
        "split": "test",
        "is_counterfactual": True,
        "degradation_type": degradation_type,
        "severity": severity,
        "degradation_params": degradation_params,
        "original_record": compact_original_record(record),
    }
    if "answer" in record:
        result["answer"] = record["answer"]
    if "source_dataset" in record:
        result["source_dataset"] = record["source_dataset"]
    return result


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_path(args.input_jsonl)
    output_jsonl = resolve_path(args.output_jsonl)
    output_image_dir = ensure_dir(resolve_path(args.output_image_dir))

    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Output JSONL exists; pass --overwrite to replace it: {output_jsonl}")

    records = read_jsonl(input_jsonl)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    degradation_types = args.degradation_types or load_default_degradation_types()
    degradation_types = [normalize_degradation_type(item) for item in degradation_types]
    unknown_types = sorted(set(degradation_types) - set(SUPPORTED_DEGRADATION_TYPES))
    if unknown_types:
        raise ValueError(f"Unsupported degradation types after alias normalization: {unknown_types}")
    degradation_types = list(dict.fromkeys(degradation_types))

    severities = [normalize_severity(item) for item in args.severities]
    unknown_severities = sorted(set(severities) - {"low", "medium", "high"})
    if unknown_severities:
        raise ValueError(f"Unsupported severities after alias normalization: {unknown_severities}")
    severities = list(dict.fromkeys(severities))

    all_pairs = [(degradation_type, severity) for degradation_type in degradation_types for severity in severities]
    output_records: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Building degraded test"):
        if args.include_clean_pairs:
            output_records.append(build_clean_pair_record(record))
        sample_rng = random.Random(stable_seed(args.seed, str(record["sample_id"]), 0))
        candidate_pairs = all_pairs[:]
        sample_rng.shuffle(candidate_pairs)
        selected_pairs = candidate_pairs[: max(0, args.num_degraded_per_sample)]
        clean_image_path = resolve_path(str(record["image_path"]))
        image = load_image(clean_image_path)

        for variant_index, (degradation_type, severity) in enumerate(selected_pairs):
            variant_rng = random.Random(stable_seed(args.seed, str(record["sample_id"]), variant_index + 1))
            degraded, params = apply_degradation_with_params(image, degradation_type, severity, variant_rng)
            output_name = f"{record['sample_id']}__deg__{degradation_type}__{severity}__{variant_index}.png"
            output_image_path = output_image_dir / output_name
            if output_image_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output image exists; pass --overwrite to replace it: {output_image_path}")
            degraded.save(output_image_path)
            params = {
                **params,
                "seed": stable_seed(args.seed, str(record["sample_id"]), variant_index + 1),
                "input_size": list(image.size),
                "output_format": "png",
            }
            output_records.append(
                build_degraded_record(
                    record=record,
                    output_image_path=output_image_path,
                    degradation_type=degradation_type,
                    severity=severity,
                    variant_index=variant_index,
                    degradation_params=params,
                )
            )

    write_jsonl(output_jsonl, output_records)
    type_distribution = Counter(record.get("degradation_type", "none") for record in output_records)
    severity_distribution = Counter(record.get("severity", "none") for record in output_records)
    stats = {
        "clean_input_count": len(records),
        "degraded_output_count": sum(1 for record in output_records if record.get("is_counterfactual")),
        "total_output_count": len(output_records),
        "degradation_type_distribution": dict(type_distribution),
        "severity_distribution": dict(severity_distribution),
        "output_jsonl": str(output_jsonl),
        "output_image_dir": str(output_image_dir),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
