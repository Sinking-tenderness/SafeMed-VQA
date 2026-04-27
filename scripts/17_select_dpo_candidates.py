from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, read_jsonl, write_json, write_jsonl


DEFAULT_QUOTA = {
    "degraded_teacher_abstain": 360,
    "clean_teacher_answer": 200,
    "degraded_teacher_answer": 160,
    "clean_teacher_abstain": 80,
}
BUCKET_PRIORITY = [
    "degraded_teacher_abstain",
    "clean_teacher_answer",
    "degraded_teacher_answer",
    "clean_teacher_abstain",
]
SAFETY_DEGRADATIONS = {"center_mask", "local_occlusion", "border_truncate", "random_crop"}
HIGHER_RISK_SEVERITIES = {"medium", "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select train-only DPO-lite candidate samples.")
    parser.add_argument("--input-jsonl", type=str, default="data/processed/train_safemed_pro.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="data/processed/dpo_lite_candidates_800.jsonl")
    parser.add_argument("--summary-json", type=str, default="outputs/dpo_lite/dpo_candidate_summary.json")
    parser.add_argument("--num-candidates", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exclude-sample-ids-jsonl", type=str, default=None)
    parser.add_argument("--max-per-clean-id", type=int, default=1)
    return parser.parse_args()


def resolve_path(path_text: str | Path | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def clean_sample_id(record: dict[str, Any]) -> str:
    return str(record.get("clean_sample_id") or record.get("source_sample_id") or record.get("sample_id"))


def clean_degraded_bucket(record: dict[str, Any]) -> str:
    if bool(record.get("is_counterfactual", False)):
        return "degraded"
    if str(record.get("degradation_type", "none")) != "none":
        return "degraded"
    return "clean"


def teacher_decision(record: dict[str, Any]) -> str | None:
    teacher_output = record.get("teacher_output")
    if not isinstance(teacher_output, dict):
        return None
    decision = str(teacher_output.get("decision", "")).strip().lower()
    return decision if decision in {"answer", "abstain"} else None


def candidate_bucket(record: dict[str, Any]) -> str | None:
    decision = teacher_decision(record)
    if decision is None:
        return None
    return f"{clean_degraded_bucket(record)}_teacher_{decision}"


def scaled_quota(num_candidates: int) -> dict[str, int]:
    if num_candidates == sum(DEFAULT_QUOTA.values()):
        return dict(DEFAULT_QUOTA)
    total = sum(DEFAULT_QUOTA.values())
    raw = {bucket: DEFAULT_QUOTA[bucket] * num_candidates / total for bucket in BUCKET_PRIORITY}
    quota = {bucket: int(raw[bucket]) for bucket in BUCKET_PRIORITY}
    remaining = num_candidates - sum(quota.values())
    fractions = sorted(
        BUCKET_PRIORITY,
        key=lambda bucket: (raw[bucket] - quota[bucket], -BUCKET_PRIORITY.index(bucket)),
        reverse=True,
    )
    for bucket in fractions[:remaining]:
        quota[bucket] += 1
    return quota


def load_excluded_sample_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    excluded: set[str] = set()
    for record in read_jsonl(path):
        sample_id = record.get("sample_id")
        if sample_id is not None:
            excluded.add(str(sample_id))
    return excluded


def sort_records_for_bucket(records: list[dict[str, Any]], bucket: str, rng: random.Random) -> list[dict[str, Any]]:
    decorated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for record in records:
        if bucket == "degraded_teacher_abstain":
            degradation_type = str(record.get("degradation_type", "none"))
            severity = str(record.get("severity", "none"))
            key = (
                0 if degradation_type in SAFETY_DEGRADATIONS else 1,
                0 if severity in HIGHER_RISK_SEVERITIES else 1,
                rng.random(),
            )
        else:
            key = (rng.random(),)
        decorated.append((key, record))
    return [record for _, record in sorted(decorated, key=lambda item: item[0])]


def select_from_bucket(
    *,
    bucket_records: list[dict[str, Any]],
    needed: int,
    selected_sample_ids: set[str],
    clean_id_counts: Counter[str],
    max_per_clean_id: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if needed <= 0:
        return selected
    for record in bucket_records:
        sample_id = str(record.get("sample_id"))
        if sample_id in selected_sample_ids:
            continue
        clean_id = clean_sample_id(record)
        if max_per_clean_id is not None and clean_id_counts[clean_id] >= max_per_clean_id:
            continue
        selected.append(record)
        selected_sample_ids.add(sample_id)
        clean_id_counts[clean_id] += 1
        if len(selected) >= needed:
            break
    return selected


def build_summary(
    *,
    input_count: int,
    selected: list[dict[str, Any]],
    quota: dict[str, int],
    missing_image_count: int,
    skipped_no_teacher_count: int,
    skipped_excluded_count: int,
    output_jsonl: Path,
    seed: int,
) -> dict[str, Any]:
    teacher_counts = Counter(teacher_decision(record) or "invalid" for record in selected)
    clean_counts = Counter(clean_degraded_bucket(record) for record in selected)
    clean_id_counts = Counter(clean_sample_id(record) for record in selected)
    return {
        "input_count": input_count,
        "selected_count": len(selected),
        "quota": quota,
        "selected_bucket_counts": dict(Counter(str(record.get("dpo_candidate_bucket", "unknown")) for record in selected)),
        "teacher_decision_counts": dict(teacher_counts),
        "clean_degraded_counts": dict(clean_counts),
        "degradation_type_counts": dict(Counter(str(record.get("degradation_type", "none")) for record in selected)),
        "severity_counts": dict(Counter(str(record.get("severity", "none")) for record in selected)),
        "missing_image_count": missing_image_count,
        "skipped_no_teacher_count": skipped_no_teacher_count,
        "skipped_excluded_count": skipped_excluded_count,
        "unique_clean_ids": len(clean_id_counts),
        "clean_id_duplicate_stats": {
            "max_records_per_clean_id": max(clean_id_counts.values(), default=0),
            "clean_id_with_multiple_records": sum(1 for count in clean_id_counts.values() if count > 1),
        },
        "output_jsonl": str(output_jsonl),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_jsonl)
    output_path = resolve_path(args.output_jsonl)
    summary_path = resolve_path(args.summary_json)
    if input_path is None or output_path is None or summary_path is None:
        raise ValueError("Input, output, and summary paths are required.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}. Pass --overwrite to replace it.")

    records = read_jsonl(input_path)
    excluded_sample_ids = load_excluded_sample_ids(resolve_path(args.exclude_sample_ids_jsonl))
    rng = random.Random(args.seed)
    quota = scaled_quota(args.num_candidates)

    missing_image_count = 0
    skipped_no_teacher_count = 0
    skipped_excluded_count = 0
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id in excluded_sample_ids:
            skipped_excluded_count += 1
            continue
        bucket = candidate_bucket(record)
        if bucket is None:
            skipped_no_teacher_count += 1
            continue
        image_path = record.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            missing_image_count += 1
            continue
        candidate = dict(record)
        candidate["dpo_candidate_bucket"] = bucket
        bucketed[bucket].append(candidate)

    sorted_bucketed = {
        bucket: sort_records_for_bucket(bucketed.get(bucket, []), bucket, rng) for bucket in BUCKET_PRIORITY
    }

    selected: list[dict[str, Any]] = []
    selected_sample_ids: set[str] = set()
    clean_id_counts: Counter[str] = Counter()
    max_per_clean_id = max(1, int(args.max_per_clean_id)) if args.max_per_clean_id else None

    for bucket in BUCKET_PRIORITY:
        selected.extend(
            select_from_bucket(
                bucket_records=sorted_bucketed[bucket],
                needed=quota[bucket],
                selected_sample_ids=selected_sample_ids,
                clean_id_counts=clean_id_counts,
                max_per_clean_id=max_per_clean_id,
            )
        )

    remaining = args.num_candidates - len(selected)
    for bucket in BUCKET_PRIORITY:
        if remaining <= 0:
            break
        added = select_from_bucket(
            bucket_records=sorted_bucketed[bucket],
            needed=remaining,
            selected_sample_ids=selected_sample_ids,
            clean_id_counts=clean_id_counts,
            max_per_clean_id=max_per_clean_id,
        )
        selected.extend(added)
        remaining = args.num_candidates - len(selected)

    for bucket in BUCKET_PRIORITY:
        if remaining <= 0:
            break
        added = select_from_bucket(
            bucket_records=sorted_bucketed[bucket],
            needed=remaining,
            selected_sample_ids=selected_sample_ids,
            clean_id_counts=clean_id_counts,
            max_per_clean_id=None,
        )
        selected.extend(added)
        remaining = args.num_candidates - len(selected)

    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)
    write_jsonl(output_path, selected)
    summary = build_summary(
        input_count=len(records),
        selected=selected,
        quota=quota,
        missing_image_count=missing_image_count,
        skipped_no_teacher_count=skipped_no_teacher_count,
        skipped_excluded_count=skipped_excluded_count,
        output_jsonl=output_path,
        seed=args.seed,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
