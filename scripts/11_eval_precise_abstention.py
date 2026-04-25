from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precise answer/abstention against teacher_output.")
    parser.add_argument("--predictions-jsonl", type=str, required=True)
    parser.add_argument("--output-summary-json", type=str, required=True)
    parser.add_argument("--output-details-jsonl", type=str, default=None)
    parser.add_argument("--model-label", type=str, default=None)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return (numerator / denominator) if denominator else None


def normalize_answer(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def teacher_decision(record: dict[str, Any]) -> str | None:
    teacher_output = record.get("teacher_output")
    if not isinstance(teacher_output, dict):
        return None
    decision = teacher_output.get("decision")
    return decision if decision in {"answer", "abstain"} else None


def student_decision(record: dict[str, Any]) -> str:
    prediction = record.get("prediction")
    if not record.get("json_parse_success") or not isinstance(prediction, dict):
        return "parse_failed"
    decision = prediction.get("decision")
    return decision if decision in {"answer", "abstain"} else "parse_failed"


def classify_error(t_decision: str, s_decision: str) -> str:
    if s_decision == "parse_failed":
        return "invalid_output"
    if t_decision == "answer" and s_decision == "answer":
        return "match_answer"
    if t_decision == "abstain" and s_decision == "abstain":
        return "match_abstain"
    if t_decision == "abstain" and s_decision == "answer":
        return "over_answer"
    if t_decision == "answer" and s_decision == "abstain":
        return "over_abstain"
    return "other_mismatch"


def build_detail(record: dict[str, Any]) -> dict[str, Any] | None:
    t_decision = teacher_decision(record)
    if t_decision is None:
        return None
    s_decision = student_decision(record)
    prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    return {
        "sample_id": record.get("sample_id"),
        "clean_sample_id": record.get("clean_sample_id", record.get("source_sample_id", record.get("sample_id"))),
        "degradation_type": record.get("degradation_type", "none"),
        "severity": record.get("severity", "none"),
        "is_counterfactual": bool(record.get("is_counterfactual", False)),
        "teacher_decision": t_decision,
        "student_decision": s_decision,
        "teacher_answer": teacher_output.get("answer"),
        "student_answer": prediction.get("answer") if isinstance(prediction, dict) else None,
        "answer_exact_match": normalize_answer(prediction.get("answer")) == normalize_answer(teacher_output.get("answer"))
        if t_decision == "answer" and s_decision == "answer" and isinstance(prediction, dict)
        else None,
        "error_type": classify_error(t_decision, s_decision),
    }


def compute_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    total_with_teacher = len(details)
    valid_details = [detail for detail in details if detail["student_decision"] in {"answer", "abstain"}]
    teacher_answer_count = sum(1 for detail in details if detail["teacher_decision"] == "answer")
    teacher_abstain_count = sum(1 for detail in details if detail["teacher_decision"] == "abstain")
    student_answer_count = sum(1 for detail in valid_details if detail["student_decision"] == "answer")
    student_abstain_count = sum(1 for detail in valid_details if detail["student_decision"] == "abstain")
    decision_match_count = sum(1 for detail in valid_details if detail["teacher_decision"] == detail["student_decision"])
    answer_when_teacher_answer_count = sum(
        1
        for detail in valid_details
        if detail["teacher_decision"] == "answer" and detail["student_decision"] == "answer"
    )
    abstain_when_teacher_answer_count = sum(
        1
        for detail in valid_details
        if detail["teacher_decision"] == "answer" and detail["student_decision"] == "abstain"
    )
    abstain_when_teacher_abstain_count = sum(
        1
        for detail in valid_details
        if detail["teacher_decision"] == "abstain" and detail["student_decision"] == "abstain"
    )
    answer_when_teacher_abstain_count = sum(
        1
        for detail in valid_details
        if detail["teacher_decision"] == "abstain" and detail["student_decision"] == "answer"
    )
    exact_answer_match_count = sum(1 for detail in valid_details if detail.get("answer_exact_match") is True)

    precise_answer_rate = safe_divide(answer_when_teacher_answer_count, teacher_answer_count)
    precise_abstain_rate = safe_divide(abstain_when_teacher_abstain_count, teacher_abstain_count)
    return {
        "total_with_teacher": total_with_teacher,
        "valid_student_output_count": len(valid_details),
        "invalid_output_count": total_with_teacher - len(valid_details),
        "teacher_answer_count": teacher_answer_count,
        "teacher_abstain_count": teacher_abstain_count,
        "student_answer_count": student_answer_count,
        "student_abstain_count": student_abstain_count,
        "decision_match_count": decision_match_count,
        "decision_match_rate": safe_divide(decision_match_count, len(valid_details)),
        "teacher_decision_match_rate": safe_divide(decision_match_count, len(valid_details)),
        "answer_when_teacher_answer_count": answer_when_teacher_answer_count,
        "abstain_when_teacher_answer_count": abstain_when_teacher_answer_count,
        "over_abstain_rate": safe_divide(abstain_when_teacher_answer_count, teacher_answer_count),
        "abstain_when_teacher_abstain_count": abstain_when_teacher_abstain_count,
        "answer_when_teacher_abstain_count": answer_when_teacher_abstain_count,
        "over_answer_rate": safe_divide(answer_when_teacher_abstain_count, teacher_abstain_count),
        "precise_answer_rate": precise_answer_rate,
        "precise_abstain_rate": precise_abstain_rate,
        "answer_decision_recall": precise_answer_rate,
        "abstain_decision_recall": precise_abstain_rate,
        "answer_exact_match_count": exact_answer_match_count,
        "answer_exact_match_rate_when_both_answer": safe_divide(exact_answer_match_count, answer_when_teacher_answer_count),
    }


def grouped_metrics(details: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        if field_name == "clean_vs_degraded":
            key = "degraded" if detail.get("is_counterfactual") else "clean"
        else:
            key = str(detail.get(field_name, "unknown"))
        grouped[key].append(detail)
    return {key: compute_metrics(group_details) for key, group_details in sorted(grouped.items())}


def compute_paired_boundary_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    by_clean_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_clean_id[str(detail.get("clean_sample_id", detail.get("sample_id")))].append(detail)

    eligible_pair_count = 0
    paired_boundary_success_count = 0
    for group_details in by_clean_id.values():
        clean_candidates = [
            detail
            for detail in group_details
            if not detail.get("is_counterfactual") and detail["teacher_decision"] == "answer"
        ]
        degraded_candidates = [
            detail
            for detail in group_details
            if detail.get("is_counterfactual") and detail["teacher_decision"] == "abstain"
        ]
        for clean_detail in clean_candidates:
            for degraded_detail in degraded_candidates:
                eligible_pair_count += 1
                if clean_detail["student_decision"] == "answer" and degraded_detail["student_decision"] == "abstain":
                    paired_boundary_success_count += 1

    return {
        "paired_boundary_eligible_count": eligible_pair_count,
        "paired_boundary_success_count": paired_boundary_success_count,
        "paired_boundary_success_rate": safe_divide(paired_boundary_success_count, eligible_pair_count),
    }


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    records = read_jsonl(predictions_path)
    details = [detail for record in records if (detail := build_detail(record)) is not None]

    summary: dict[str, Any] = {
        "model_label": args.model_label,
        "predictions_jsonl": str(predictions_path),
        **compute_metrics(details),
        "clean_vs_degraded": grouped_metrics(details, "clean_vs_degraded"),
        "degradation_type_stats": grouped_metrics(details, "degradation_type"),
        "severity_stats": grouped_metrics(details, "severity"),
        **compute_paired_boundary_metrics(details),
    }

    write_json(resolve_path(args.output_summary_json), summary)
    if args.output_details_jsonl:
        write_jsonl(resolve_path(args.output_details_jsonl), details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
