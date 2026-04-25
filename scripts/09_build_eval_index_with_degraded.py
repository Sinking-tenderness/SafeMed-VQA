from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge clean and degraded test JSONL into a paired eval index.")
    parser.add_argument("--clean-jsonl", type=str, default="data/interim/test_index.jsonl")
    parser.add_argument("--degraded-jsonl", type=str, default="data/interim/test_degraded_index.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="data/interim/test_paired_clean_degraded_index.jsonl")
    parser.add_argument("--max-clean-samples", type=int, default=None)
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-degraded", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def normalize_clean_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["clean_sample_id"] = record["sample_id"]
    result["clean_image_path"] = record["image_path"]
    result["reference_answer"] = record.get("reference_answer", record.get("answer", ""))
    result["is_counterfactual"] = False
    result["degradation_type"] = "none"
    result["severity"] = "none"
    result["split"] = "test"
    return result


def normalize_degraded_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["clean_sample_id"] = record.get("clean_sample_id", record.get("source_sample_id", record.get("sample_id")))
    result["clean_image_path"] = record.get("clean_image_path", record.get("image_path"))
    result["reference_answer"] = record.get("reference_answer", record.get("answer", ""))
    result["is_counterfactual"] = True
    result["degradation_type"] = record.get("degradation_type", "unknown")
    result["severity"] = record.get("severity", "unknown")
    result["split"] = "test"
    return result


def main() -> None:
    args = parse_args()
    clean_records = read_jsonl(resolve_path(args.clean_jsonl))
    if args.max_clean_samples is not None:
        allowed_clean_ids = {record["sample_id"] for record in clean_records[: args.max_clean_samples]}
        clean_records = clean_records[: args.max_clean_samples]
    else:
        allowed_clean_ids = {record["sample_id"] for record in clean_records}

    output_records: list[dict[str, Any]] = []
    if args.include_clean:
        for record in tqdm(clean_records, desc="Adding clean records"):
            output_records.append(normalize_clean_record(record))

    if args.include_degraded:
        degraded_records = read_jsonl(resolve_path(args.degraded_jsonl))
        for record in tqdm(degraded_records, desc="Adding degraded records"):
            clean_sample_id = record.get("clean_sample_id", record.get("source_sample_id"))
            if args.max_clean_samples is not None and clean_sample_id not in allowed_clean_ids:
                continue
            output_records.append(normalize_degraded_record(record))

    output_jsonl = resolve_path(args.output_jsonl)
    write_jsonl(output_jsonl, output_records)

    type_distribution = Counter(record.get("degradation_type", "unknown") for record in output_records)
    severity_distribution = Counter(record.get("severity", "unknown") for record in output_records)
    stats = {
        "total_samples": len(output_records),
        "clean_count": sum(1 for record in output_records if not record.get("is_counterfactual")),
        "degraded_count": sum(1 for record in output_records if record.get("is_counterfactual")),
        "degradation_type_distribution": dict(type_distribution),
        "severity_distribution": dict(severity_distribution),
        "output_jsonl": str(output_jsonl),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
