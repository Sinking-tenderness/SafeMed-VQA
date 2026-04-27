from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, read_jsonl, write_json, write_jsonl


PAIR_QUOTA = {
    "over_answer": 90,
    "wrong_answer": 55,
    "over_abstain": 40,
    "invalid_schema": 15,
}
PAIR_PRIORITY = ["over_answer", "wrong_answer", "over_abstain", "invalid_schema"]
BASE_PRIORITY = {
    "over_answer": 100,
    "wrong_answer": 80,
    "over_abstain": 70,
    "invalid_schema": 60,
}
SAFETY_DEGRADATIONS = {"center_mask", "local_occlusion", "border_truncate", "random_crop"}
HIGHER_RISK_SEVERITIES = {"medium", "high"}
ABSTAIN_FALLBACK = "I cannot answer safely because the image evidence is insufficient."
INVALID_REJECTED_FALLBACK = {
    "explanation": "Invalid output.",
    "raw_confidence": 0.0,
    "decision": "answer",
    "abstain_type": "none",
    "risk_level": "high",
    "answer": "",
}
FORBIDDEN_INPUT_MARKERS = {
    "test_paired_teacher_labeled",
    "eval_models_teacher_labeled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO-lite preference pairs from train-only candidate predictions.")
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="outputs/dpo_lite_candidate_eval/sft_lora_transformers/predictions.jsonl",
    )
    parser.add_argument(
        "--answer-correctness-jsonl",
        type=str,
        default="outputs/dpo_lite_candidate_eval/sft_lora_transformers/answer_correctness_details.jsonl",
    )
    parser.add_argument("--output-all-jsonl", type=str, default="data/processed/dpo_lite_pairs_all.jsonl")
    parser.add_argument("--output-train-jsonl", type=str, default="data/processed/dpo_lite_pairs_train.jsonl")
    parser.add_argument("--output-val-jsonl", type=str, default="data/processed/dpo_lite_pairs_val.jsonl")
    parser.add_argument("--summary-json", type=str, default="outputs/dpo_lite/dpo_pair_summary.json")
    parser.add_argument("--target-pairs", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partially-correct-as-negative", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str | Path | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def guard_train_only_input(path: Path) -> None:
    lowered = str(path).lower()
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in lowered:
            raise ValueError(f"Refusing to build DPO training data from forbidden test/eval path: {path}")


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


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


def student_decision(record: dict[str, Any]) -> str:
    prediction = record.get("prediction")
    if not bool(record.get("json_parse_success", False)) or not isinstance(prediction, dict):
        return "parse_failed"
    decision = str(prediction.get("decision", "")).strip().lower()
    return decision if decision in {"answer", "abstain"} else "other"


def compact_for_yesno(text: Any) -> str:
    normalized = normalize_text(text)
    compact = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(compact.split())


def strict_yes_no_label(text: Any) -> str | None:
    compact = compact_for_yesno(text)
    if not compact:
        return None
    negative_patterns = [
        r"^no(?:\b| )",
        r"^false(?:\b| )",
        r"^absent(?:\b| )",
        r"\bno evidence(?: of)?\b",
        r"\bnot present\b",
        r"\bnot visible\b",
        r"\bnot seen\b",
        r"\bnegative for\b",
    ]
    positive_patterns = [
        r"^yes(?:\b| )",
        r"^true(?:\b| )",
        r"^present(?:\b| )",
        r"\bevidence of\b",
        r"\bvisible\b",
        r"\bseen\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, compact):
            return "no"
    for pattern in positive_patterns:
        if re.search(pattern, compact):
            return "yes"
    return None


def yes_no_conflict(left: Any, right: Any) -> bool:
    left_label = strict_yes_no_label(left)
    right_label = strict_yes_no_label(right)
    return left_label is not None and right_label is not None and left_label != right_label


def reference_teacher_yesno_conflict(record: dict[str, Any]) -> bool:
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    reference_answer = str(record.get("reference_answer") or "").strip()
    teacher_answer = str(teacher_output.get("answer") or "").strip()
    return bool(reference_answer and teacher_answer and yes_no_conflict(reference_answer, teacher_answer))


def student_yesno_wrong(record: dict[str, Any]) -> bool:
    prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
    gold_answer = str(record.get("reference_answer") or "").strip()
    if not gold_answer:
        teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
        gold_answer = str(teacher_output.get("answer") or "").strip()
    student_answer = str(prediction.get("answer") or "").strip()
    return bool(gold_answer and student_answer and yes_no_conflict(gold_answer, student_answer))


def load_answer_correctness(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    details: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = record.get("sample_id")
        if sample_id is not None:
            details[str(sample_id)] = record
    return details


def strict_json_string(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_chosen_output(record: dict[str, Any]) -> dict[str, Any] | None:
    teacher_output = record.get("teacher_output")
    if not isinstance(teacher_output, dict):
        return None
    decision = teacher_decision(record)
    if decision is None:
        return None

    chosen = dict(teacher_output)
    chosen["decision"] = decision
    chosen["explanation"] = str(chosen.get("explanation") or "Teacher preference output.").strip()
    if decision == "abstain":
        chosen["abstain_type"] = str(chosen.get("abstain_type") or "").strip().lower()
        if chosen["abstain_type"] in {"", "none"}:
            risk_level = str(chosen.get("risk_level") or "").strip().lower()
            chosen["abstain_type"] = "high_risk_uncertainty" if risk_level == "high" else "visual_insufficiency"
        chosen["answer"] = str(chosen.get("answer") or "").strip() or ABSTAIN_FALLBACK
        chosen["risk_level"] = str(chosen.get("risk_level") or "high").strip().lower() or "high"
        chosen["raw_confidence"] = float(chosen.get("raw_confidence", 0.0) or 0.0)
        return chosen

    answer = str(record.get("reference_answer") or "").strip()
    if not answer:
        answer = str(chosen.get("answer") or "").strip()
    if not answer:
        return None
    chosen["answer"] = answer
    chosen["abstain_type"] = "none"
    chosen["risk_level"] = str(chosen.get("risk_level") or "low").strip().lower() or "low"
    chosen["raw_confidence"] = float(chosen.get("raw_confidence", 1.0) or 1.0)
    return chosen


def build_rejected(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    prediction = record.get("prediction")
    if bool(record.get("json_parse_success", False)) and isinstance(prediction, dict):
        return strict_json_string(prediction), prediction
    raw_output = str(record.get("raw_output") or "").strip()
    if raw_output:
        return raw_output, None
    return strict_json_string(INVALID_REJECTED_FALLBACK), None


def answer_correctness_outcome(
    record: dict[str, Any],
    correctness_by_id: dict[str, dict[str, Any]],
) -> str | None:
    sample_id = str(record.get("sample_id"))
    detail = correctness_by_id.get(sample_id)
    if detail is not None:
        outcome = detail.get("outcome")
        return str(outcome) if outcome is not None else None
    if student_yesno_wrong(record):
        return "wrong"
    return None


def boundary_failure_sample_ids(records: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[clean_sample_id(record)].append(record)

    marked: set[str] = set()
    for group_records in grouped.values():
        clean_failures = [
            str(record.get("sample_id"))
            for record in group_records
            if clean_degraded_bucket(record) == "clean"
            and teacher_decision(record) == "answer"
            and student_decision(record) != "answer"
        ]
        degraded_failures = [
            str(record.get("sample_id"))
            for record in group_records
            if clean_degraded_bucket(record) == "degraded"
            and teacher_decision(record) == "abstain"
            and student_decision(record) == "answer"
        ]
        if clean_failures and degraded_failures:
            marked.update(clean_failures)
            marked.update(degraded_failures)
    return marked


def priority_score(pair_type: str, record: dict[str, Any], prediction: dict[str, Any] | None, boundary_failure: bool) -> float:
    score = float(BASE_PRIORITY[pair_type])
    degradation_type = str(record.get("degradation_type", "none"))
    severity = str(record.get("severity", "none"))
    if boundary_failure:
        score += 20
    if degradation_type in SAFETY_DEGRADATIONS:
        score += 10
    if severity in HIGHER_RISK_SEVERITIES:
        score += 8
    if isinstance(prediction, dict):
        try:
            confidence = float(prediction.get("raw_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= 0.8:
            score += 10
        elif confidence >= 0.65:
            score += 5
        if str(prediction.get("risk_level", "")).strip().lower() == "high":
            score += 5
    if clean_degraded_bucket(record) == "degraded" and teacher_decision(record) == "abstain":
        score += 5
    return score


def infer_pair_type(
    record: dict[str, Any],
    correctness_by_id: dict[str, dict[str, Any]],
    allow_partially_correct: bool,
) -> tuple[str | None, str | None]:
    t_decision = teacher_decision(record)
    s_decision = student_decision(record)
    if s_decision == "parse_failed":
        return "invalid_schema", "invalid_output"
    if t_decision == "abstain" and s_decision == "answer":
        return "over_answer", None
    if t_decision == "answer" and s_decision == "abstain":
        return "over_abstain", "over_abstain"
    if t_decision == "answer" and s_decision == "answer":
        outcome = answer_correctness_outcome(record, correctness_by_id)
        if outcome == "wrong":
            return "wrong_answer", outcome
        if allow_partially_correct and outcome == "partially_correct":
            return "wrong_answer", outcome
        return None, outcome
    return None, None


def build_pair(
    record: dict[str, Any],
    *,
    pair_type: str,
    answer_outcome: str | None,
    boundary_failure: bool,
) -> dict[str, Any] | None:
    chosen_output = build_chosen_output(record)
    if chosen_output is None:
        return None
    chosen = strict_json_string(chosen_output)
    rejected, rejected_output = build_rejected(record)
    if chosen.strip() == rejected.strip():
        return None

    prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else None
    clean_id = clean_sample_id(record)
    sample_id = str(record.get("sample_id"))
    score = priority_score(pair_type, record, prediction, boundary_failure)
    if answer_outcome == "partially_correct":
        score -= 10

    return {
        "pair_id": f"dpo_lite::{pair_type}::{sample_id}",
        "sample_id": sample_id,
        "clean_sample_id": clean_id,
        "image_path": record.get("image_path"),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "teacher_output": record.get("teacher_output"),
        "chosen": chosen,
        "rejected": rejected,
        "chosen_output": chosen_output,
        "rejected_output": rejected_output,
        "raw_rejected_output": record.get("raw_output"),
        "pair_type": pair_type,
        "boundary_failure": boundary_failure,
        "degradation_type": record.get("degradation_type", "none"),
        "severity": record.get("severity", "none"),
        "is_counterfactual": bool(record.get("is_counterfactual", False)),
        "teacher_decision": teacher_decision(record),
        "student_decision": student_decision(record),
        "answer_correctness_outcome": answer_outcome,
        "priority_score": score,
    }


def scaled_quota(target_pairs: int) -> dict[str, int]:
    default_total = sum(PAIR_QUOTA.values())
    if target_pairs == default_total:
        return dict(PAIR_QUOTA)
    raw = {pair_type: PAIR_QUOTA[pair_type] * target_pairs / default_total for pair_type in PAIR_PRIORITY}
    quota = {pair_type: int(raw[pair_type]) for pair_type in PAIR_PRIORITY}
    remaining = target_pairs - sum(quota.values())
    fractions = sorted(
        PAIR_PRIORITY,
        key=lambda pair_type: (raw[pair_type] - quota[pair_type], -PAIR_PRIORITY.index(pair_type)),
        reverse=True,
    )
    for pair_type in fractions[:remaining]:
        quota[pair_type] += 1
    return quota


def select_pairs(candidate_pairs: list[dict[str, Any]], target_pairs: int, rng: random.Random) -> list[dict[str, Any]]:
    quota = scaled_quota(target_pairs)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tie_breakers = {pair["pair_id"]: rng.random() for pair in candidate_pairs}
    for pair in candidate_pairs:
        by_type[str(pair["pair_type"])].append(pair)
    for pair_type in PAIR_PRIORITY:
        by_type[pair_type].sort(key=lambda pair: (-float(pair["priority_score"]), tie_breakers[pair["pair_id"]]))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    clean_counts: Counter[str] = Counter()

    def add_from_type(pair_type: str, needed: int, max_per_clean_id: int | None) -> None:
        if needed <= 0:
            return
        for pair in by_type[pair_type]:
            if pair["pair_id"] in selected_ids:
                continue
            clean_id = str(pair["clean_sample_id"])
            if max_per_clean_id is not None and clean_counts[clean_id] >= max_per_clean_id:
                continue
            selected.append(pair)
            selected_ids.add(pair["pair_id"])
            clean_counts[clean_id] += 1
            if len(selected) >= target_pairs or needed <= 1:
                break
            needed -= 1

    for pair_type in PAIR_PRIORITY:
        add_from_type(pair_type, quota[pair_type], 2)

    remaining = target_pairs - len(selected)
    for pair_type in PAIR_PRIORITY:
        if remaining <= 0:
            break
        before = len(selected)
        add_from_type(pair_type, remaining, 2)
        remaining -= len(selected) - before

    for pair_type in PAIR_PRIORITY:
        if remaining <= 0:
            break
        before = len(selected)
        add_from_type(pair_type, remaining, None)
        remaining -= len(selected) - before

    selected.sort(key=lambda pair: (-float(pair["priority_score"]), PAIR_PRIORITY.index(str(pair["pair_type"])), pair["pair_id"]))
    return selected[:target_pairs]


def split_train_val(
    selected_pairs: list[dict[str, Any]],
    *,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in selected_pairs:
        grouped[str(pair["clean_sample_id"])].append(pair)
    clean_ids = list(grouped)
    rng.shuffle(clean_ids)
    target_val = int(round(len(selected_pairs) * max(0.0, min(val_ratio, 0.5))))
    if target_val > 0:
        target_val = max(1, target_val)

    val_clean_ids: set[str] = set()
    val_count = 0
    for clean_id in clean_ids:
        if val_count >= target_val:
            break
        val_clean_ids.add(clean_id)
        val_count += len(grouped[clean_id])

    train_pairs: list[dict[str, Any]] = []
    val_pairs: list[dict[str, Any]] = []
    for pair in selected_pairs:
        if str(pair["clean_sample_id"]) in val_clean_ids:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)
    return train_pairs, val_pairs


def ensure_outputs_can_write(paths: list[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")


def summarize(
    *,
    prediction_count: int,
    candidate_pairs: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    skipped_counts: Counter[str],
    output_all_jsonl: Path,
    output_train_jsonl: Path,
    output_val_jsonl: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        "input_prediction_count": prediction_count,
        "candidate_pair_count": len(candidate_pairs),
        "selected_pair_count": len(selected_pairs),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "available_pair_type_counts": dict(Counter(str(pair["pair_type"]) for pair in candidate_pairs)),
        "selected_pair_type_counts": dict(Counter(str(pair["pair_type"]) for pair in selected_pairs)),
        "boundary_failure_selected_count": sum(1 for pair in selected_pairs if bool(pair.get("boundary_failure", False))),
        "teacher_decision_counts": dict(Counter(str(pair.get("teacher_decision")) for pair in selected_pairs)),
        "student_decision_counts": dict(Counter(str(pair.get("student_decision")) for pair in selected_pairs)),
        "degradation_type_counts": dict(Counter(str(pair.get("degradation_type", "none")) for pair in selected_pairs)),
        "severity_counts": dict(Counter(str(pair.get("severity", "none")) for pair in selected_pairs)),
        "skipped_counts": dict(skipped_counts),
        "output_all_jsonl": str(output_all_jsonl),
        "output_train_jsonl": str(output_train_jsonl),
        "output_val_jsonl": str(output_val_jsonl),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    answer_correctness_path = resolve_path(args.answer_correctness_jsonl)
    output_all_path = resolve_path(args.output_all_jsonl)
    output_train_path = resolve_path(args.output_train_jsonl)
    output_val_path = resolve_path(args.output_val_jsonl)
    summary_path = resolve_path(args.summary_json)
    paths = [predictions_path, output_all_path, output_train_path, output_val_path, summary_path]
    if any(path is None for path in paths):
        raise ValueError("All required paths must be provided.")
    assert predictions_path is not None
    assert output_all_path is not None
    assert output_train_path is not None
    assert output_val_path is not None
    assert summary_path is not None

    guard_train_only_input(predictions_path)
    ensure_outputs_can_write([output_all_path, output_train_path, output_val_path, summary_path], args.overwrite)

    records = read_jsonl(predictions_path)
    correctness_by_id = load_answer_correctness(answer_correctness_path)
    boundary_failures = boundary_failure_sample_ids(records)
    skipped_counts: Counter[str] = Counter()
    candidate_pairs: list[dict[str, Any]] = []

    for record in records:
        if str(record.get("split", "train")).lower() == "test":
            skipped_counts["test_split"] += 1
            continue
        if teacher_decision(record) is None:
            skipped_counts["missing_or_invalid_teacher_output"] += 1
            continue
        image_path = record.get("image_path")
        if not image_path or not Path(str(image_path)).exists():
            skipped_counts["missing_image"] += 1
            continue
        if reference_teacher_yesno_conflict(record):
            skipped_counts["reference_teacher_yesno_conflict"] += 1
            continue

        pair_type, answer_outcome = infer_pair_type(
            record,
            correctness_by_id,
            allow_partially_correct=args.allow_partially_correct_as_negative,
        )
        if pair_type is None:
            skipped_counts["not_preference_failure"] += 1
            continue
        pair = build_pair(
            record,
            pair_type=pair_type,
            answer_outcome=answer_outcome,
            boundary_failure=str(record.get("sample_id")) in boundary_failures,
        )
        if pair is None:
            skipped_counts["invalid_pair_payload"] += 1
            continue
        candidate_pairs.append(pair)

    rng = random.Random(args.seed)
    selected_pairs = select_pairs(candidate_pairs, args.target_pairs, rng)
    train_pairs, val_pairs = split_train_val(selected_pairs, val_ratio=args.val_ratio, rng=rng)

    for path in [output_all_path, output_train_path, output_val_path, summary_path]:
        ensure_dir(path.parent)
    write_jsonl(output_all_path, selected_pairs)
    write_jsonl(output_train_path, train_pairs)
    write_jsonl(output_val_path, val_pairs)
    summary = summarize(
        prediction_count=len(records),
        candidate_pairs=candidate_pairs,
        selected_pairs=selected_pairs,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        skipped_counts=skipped_counts,
        output_all_jsonl=output_all_path,
        output_train_jsonl=output_train_path,
        output_val_jsonl=output_val_path,
        seed=args.seed,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
