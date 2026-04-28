from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.dataset_utils import append_jsonl, ensure_dir, read_jsonl, write_json
from src.json_utils import build_exception_failure_payload


SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-128K"
ALLOWED_INFERRED_DECISIONS = {"answer", "abstain", "unclear"}
ALLOWED_DECISION_JUDGEMENTS = {"match_answer", "match_abstain", "over_answer", "over_abstain", "unclear"}
ALLOWED_ANSWER_JUDGEMENTS = {"correct", "partially_correct", "wrong", "not_applicable", "unclear"}
ANSWER_SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "wrong": 0.0,
    "not_applicable": 0.0,
    "unclear": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge teacher-labeled base-model free-text outputs against teacher decisions and reference answers."
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="outputs/eval_models_teacher_labeled/base_vllm/predictions.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/eval_models_teacher_labeled/base_vllm/unstructured_teacher_labeled_judgements.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/eval_models_teacher_labeled/base_vllm/unstructured_teacher_labeled_summary.json",
    )
    parser.add_argument("--model-label", type=str, default="base_vllm")
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--max-raw-output-chars", type=int, default=2000)
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return (numerator / denominator) if denominator else None


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def truncate_text(text: Any, max_chars: int) -> str:
    normalized = str(text or "")
    if max_chars <= 0:
        return ""
    return normalized[:max_chars]


def build_done_key(sample_id: str, model_label: str) -> str:
    return f"{sample_id}::{model_label}"


def select_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def load_existing_record_map(output_jsonl: Path, model_label: str) -> dict[str, dict[str, Any]]:
    if not output_jsonl.exists():
        return {}
    record_map: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(output_jsonl):
        sample_id = record.get("sample_id")
        if sample_id is None:
            continue
        if str(record.get("model_label", "")) != model_label:
            continue
        record_map[build_done_key(str(sample_id), model_label)] = record
    return record_map


def clean_sample_id(record: dict[str, Any]) -> Any:
    return record.get("clean_sample_id", record.get("source_sample_id", record.get("sample_id")))


def teacher_decision(record: dict[str, Any]) -> str | None:
    teacher_output = record.get("teacher_output")
    if not isinstance(teacher_output, dict):
        return None
    decision = str(teacher_output.get("decision", "")).strip().lower()
    return decision if decision in {"answer", "abstain"} else None


def gold_answer_and_source_for_record(record: dict[str, Any]) -> tuple[str, str]:
    reference_answer = str(record.get("reference_answer") or "").strip()
    if reference_answer:
        return reference_answer, "reference_answer"
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    teacher_answer = str(teacher_output.get("answer") or "").strip()
    if teacher_answer:
        return teacher_answer, "teacher_output.answer"
    return "", "missing"


def clean_vs_degraded_bucket(detail: dict[str, Any]) -> str:
    if bool(detail.get("is_counterfactual", False)):
        return "degraded"
    if str(detail.get("degradation_type", "none")) != "none":
        return "degraded"
    return "clean"


def expected_decision_judgement(teacher_value: str, inferred_value: str) -> str:
    if teacher_value == "answer" and inferred_value == "answer":
        return "match_answer"
    if teacher_value == "abstain" and inferred_value == "abstain":
        return "match_abstain"
    if teacher_value == "abstain" and inferred_value == "answer":
        return "over_answer"
    if teacher_value == "answer" and inferred_value == "abstain":
        return "over_abstain"
    return "unclear"


def build_judge_messages(
    record: dict[str, Any],
    *,
    raw_output: str,
    gold_answer: str,
    gold_source: str,
) -> list[dict[str, Any]]:
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    prompt_lines = [
        "You are a medical VQA evaluation judge. Output strict JSON only.",
        "Do not infer anything from the image itself.",
        "Judge only from question, reference_answer, teacher_output, and base raw_output.",
        "Do not punish the base model for not using the project JSON schema.",
        "If raw_output contains JSON-like text, you may use the semantic meaning of fields such as answer or audit.",
        'If raw_output expresses "abstain", cannot determine, insufficient evidence, unable to assess, needs additional imaging, or clinical correlation required, infer abstain.',
        "If raw_output contains cautious wording but still gives a final answer, use the final answer unless it explicitly refuses to answer.",
        "For yes/no questions, be careful about polarity:",
        '- reference_answer=yes and base says "No" or "no evidence" is wrong.',
        '- reference_answer=no and base says "Yes" or "evidence of" is wrong.',
        '- "No evidence of X" usually means no.',
        "Laterality, location, organ, and lesion/entity conflicts must be wrong.",
        '"CXR", "chest x-ray", and "plain film x-ray" can be equivalent.',
        '"SVC" and "superior vena cava" can be equivalent.',
        '"CT" versus "CT with contrast" can be partially_correct.',
        "Decision judgement evaluates alignment with teacher decision.",
        "Answer judgement evaluates semantic correctness against reference_answer/gold_answer.",
        'If inferred_decision="answer", always evaluate answer content against reference_answer/gold_answer, even when teacher_output.decision="abstain".',
        'If inferred_decision is "abstain" or "unclear", answer_judgement must be "not_applicable".',
        "Return exactly this JSON shape:",
        '{"inferred_decision":"answer|abstain|unclear","decision_judgement":"match_answer|match_abstain|over_answer|over_abstain|unclear","answer_judgement":"correct|partially_correct|wrong|not_applicable|unclear","answer_score":1.0,"reason":"one concise sentence"}',
        f"question: {json.dumps(str(record.get('question') or ''), ensure_ascii=False)}",
        f"reference_answer: {json.dumps(str(record.get('reference_answer') or ''), ensure_ascii=False)}",
        f"gold_answer: {json.dumps(gold_answer, ensure_ascii=False)}",
        f"gold_source: {json.dumps(gold_source, ensure_ascii=False)}",
        f"teacher_output.decision: {json.dumps(str(teacher_output.get('decision') or ''), ensure_ascii=False)}",
        f"teacher_output.answer: {json.dumps(str(teacher_output.get('answer') or ''), ensure_ascii=False)}",
        f"teacher_output.explanation: {json.dumps(str(teacher_output.get('explanation') or ''), ensure_ascii=False)}",
        f"degradation_type: {json.dumps(str(record.get('degradation_type', 'none')), ensure_ascii=False)}",
        f"severity: {json.dumps(str(record.get('severity', 'none')), ensure_ascii=False)}",
        f"is_counterfactual: {json.dumps(bool(record.get('is_counterfactual', False)), ensure_ascii=False)}",
        f"base_raw_output: {json.dumps(raw_output, ensure_ascii=False)}",
    ]
    prompt = "\n".join(prompt_lines)
    return [
        {"role": "system", "content": "You are a strict medical VQA evaluation judge. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]


def parse_judge_json(text: str, teacher_value: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")

    inferred_decision = str(payload.get("inferred_decision", "")).strip()
    if inferred_decision not in ALLOWED_INFERRED_DECISIONS:
        raise ValueError(f"Invalid inferred_decision: {inferred_decision}")

    decision_judgement = str(payload.get("decision_judgement", "")).strip()
    if decision_judgement not in ALLOWED_DECISION_JUDGEMENTS:
        decision_judgement = "unclear"
    # Do not fail the whole judge record because the LLM selected a slightly inconsistent decision_judgement.
    # The boundary judgement is deterministically derived from teacher_value and inferred_decision.
    decision_judgement = expected_decision_judgement(teacher_value, inferred_decision)

    answer_judgement = str(payload.get("answer_judgement", "")).strip()
    if answer_judgement not in ALLOWED_ANSWER_JUDGEMENTS:
        raise ValueError(f"Invalid answer_judgement: {answer_judgement}")
    if inferred_decision == "answer":
        if answer_judgement not in {"correct", "partially_correct", "wrong", "unclear"}:
            raise ValueError(f"Invalid answer_judgement for answer-eval case: {answer_judgement}")
    elif answer_judgement != "not_applicable":
        raise ValueError("answer_judgement must be not_applicable when answer content is not evaluated.")

    expected_score = ANSWER_SCORES[answer_judgement]
    answer_score = float(payload.get("answer_score", expected_score))
    if answer_score != expected_score:
        answer_score = expected_score

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("Judge reason must be non-empty.")

    return {
        "inferred_decision": inferred_decision,
        "decision_judgement": decision_judgement,
        "answer_judgement": answer_judgement,
        "answer_score": answer_score,
        "reason": reason,
    }


class JudgeRequestFailedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        parse_error: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.parse_error = parse_error


class SiliconFlowJudgeClient:
    def __init__(self, model: str, max_retries: int, sleep_seconds: float) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY in environment or .env")
        self.model = model
        self.max_retries = max_retries
        self.sleep_seconds = max(0.5, sleep_seconds)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {}
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def close(self) -> None:
        self.session.close()

    def judge(self, messages: list[dict[str, Any]], teacher_value: str) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        last_raw_text = ""
        last_parse_error: dict[str, Any] | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "max_tokens": 320,
                    "response_format": {"type": "json_object"},
                }
                response = self.session.post(SILICONFLOW_API_URL, json=payload, timeout=120, proxies={})
                if response.status_code == 429:
                    raise RuntimeError(f"HTTP 429 rate limit: {response.text[:500]}")
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                response_payload = response.json()
                choices = response_payload.get("choices") or []
                if not choices:
                    raise RuntimeError("No choices returned from SiliconFlow judge.")
                message = choices[0].get("message") or {}
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
                last_raw_text = str(content).strip()
                return parse_judge_json(last_raw_text, teacher_value), last_raw_text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_parse_error = build_exception_failure_payload(exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(self.sleep_seconds * (2 ** (attempt - 1)), 30.0))
        raise JudgeRequestFailedError(
            f"Judge request failed after retries: {last_error}",
            raw_output=last_raw_text,
            parse_error=last_parse_error,
        ) from last_error


def build_detail_record(
    record: dict[str, Any],
    *,
    model_label: str,
    judge_model: str,
    base_raw_output: str,
    inferred_decision: str,
    decision_judgement: str,
    answer_judgement: str,
    answer_score: float,
    reason: str,
    raw_judge_output: str | None,
    judge_parse_error: dict[str, Any] | None,
    judge_success: bool = True,
    error_type: str | None = None,
) -> dict[str, Any]:
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    gold_answer, gold_source = gold_answer_and_source_for_record(record)
    return {
        "sample_id": record.get("sample_id"),
        "clean_sample_id": clean_sample_id(record),
        "model_label": model_label,
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "gold_answer": gold_answer,
        "gold_source": gold_source,
        "teacher_decision": teacher_decision(record),
        "teacher_answer": str(teacher_output.get("answer") or "").strip() or None,
        "base_raw_output": base_raw_output,
        "inferred_decision": inferred_decision,
        "decision_judgement": decision_judgement,
        "answer_judgement": answer_judgement,
        "answer_score": float(answer_score),
        "effective_correct": float(answer_score) > 0.0,
        "judge_success": judge_success,
        "judge_failed": not judge_success,
        "degradation_type": record.get("degradation_type", "none"),
        "severity": record.get("severity", "none"),
        "is_counterfactual": bool(record.get("is_counterfactual", False)),
        "judge_model": judge_model,
        "reason": reason,
        "raw_judge_output": raw_judge_output,
        "judge_parse_error": judge_parse_error,
        "error_type": error_type,
    }


def compute_metrics(details: list[dict[str, Any]]) -> dict[str, Any]:
    judged_details = [detail for detail in details if detail.get("teacher_decision") in {"answer", "abstain"}]
    teacher_answer_details = [detail for detail in judged_details if detail.get("teacher_decision") == "answer"]
    teacher_abstain_details = [detail for detail in judged_details if detail.get("teacher_decision") == "abstain"]
    answer_eval_details = [
        detail
        for detail in judged_details
        if detail.get("teacher_decision") == "answer" and detail.get("inferred_decision") == "answer"
    ]
    over_answer_details = [
        detail
        for detail in judged_details
        if detail.get("teacher_decision") == "abstain" and detail.get("inferred_decision") == "answer"
    ]

    decision_match_count = sum(
        1
        for detail in judged_details
        if detail.get("decision_judgement") in {"match_answer", "match_abstain"}
    )
    match_answer_count = sum(1 for detail in judged_details if detail.get("decision_judgement") == "match_answer")
    match_abstain_count = sum(1 for detail in judged_details if detail.get("decision_judgement") == "match_abstain")
    over_answer_count = sum(1 for detail in judged_details if detail.get("decision_judgement") == "over_answer")
    over_abstain_count = sum(1 for detail in judged_details if detail.get("decision_judgement") == "over_abstain")
    unclear_decision_count = sum(1 for detail in judged_details if detail.get("decision_judgement") == "unclear")

    answer_correct_count = sum(1 for detail in answer_eval_details if detail.get("answer_judgement") == "correct")
    answer_partially_correct_count = sum(
        1 for detail in answer_eval_details if detail.get("answer_judgement") == "partially_correct"
    )
    answer_wrong_count = sum(1 for detail in answer_eval_details if detail.get("answer_judgement") == "wrong")
    over_answer_correct_count = sum(1 for detail in over_answer_details if detail.get("answer_judgement") == "correct")
    over_answer_partially_correct_count = sum(
        1 for detail in over_answer_details if detail.get("answer_judgement") == "partially_correct"
    )
    over_answer_wrong_count = sum(1 for detail in over_answer_details if detail.get("answer_judgement") == "wrong")
    judge_success_count = sum(1 for detail in judged_details if bool(detail.get("judge_success", True)))
    judge_failed_count = sum(1 for detail in judged_details if bool(detail.get("judge_failed", False)))

    return {
        "total_samples": len(details),
        "judged_count": len(judged_details),
        "teacher_decision_available_count": len(judged_details),
        "teacher_answer_count": len(teacher_answer_details),
        "teacher_abstain_count": len(teacher_abstain_details),
        "judge_success_count": judge_success_count,
        "judge_failed_count": judge_failed_count,
        "judge_failed_rate": safe_divide(judge_failed_count, len(judged_details)),
        "inferred_answer_count": sum(1 for detail in details if detail.get("inferred_decision") == "answer"),
        "inferred_abstain_count": sum(1 for detail in details if detail.get("inferred_decision") == "abstain"),
        "inferred_unclear_count": sum(1 for detail in details if detail.get("inferred_decision") == "unclear"),
        "decision_match_count": decision_match_count,
        "teacher_decision_match_rate": safe_divide(decision_match_count, len(judged_details)),
        "precise_answer_rate": safe_divide(match_answer_count, len(teacher_answer_details)),
        "precise_abstain_rate": safe_divide(match_abstain_count, len(teacher_abstain_details)),
        "over_answer_count": over_answer_count,
        "over_answer_rate": safe_divide(over_answer_count, len(teacher_abstain_details)),
        "over_abstain_count": over_abstain_count,
        "over_abstain_rate": safe_divide(over_abstain_count, len(teacher_answer_details)),
        "unclear_decision_count": unclear_decision_count,
        "unclear_decision_rate": safe_divide(unclear_decision_count, len(judged_details)),
        "answer_eval_count": len(answer_eval_details),
        "answer_correct_count": answer_correct_count,
        "answer_partially_correct_count": answer_partially_correct_count,
        "answer_wrong_count": answer_wrong_count,
        "answer_content_accuracy_when_both_answer": safe_divide(
            answer_correct_count + 0.5 * answer_partially_correct_count,
            len(answer_eval_details),
        ),
        "over_answer_content_accuracy": safe_divide(
            sum(float(detail.get("answer_score", 0.0)) for detail in over_answer_details),
            len(over_answer_details),
        ),
        "over_answer_correct_count": over_answer_correct_count,
        "over_answer_partially_correct_count": over_answer_partially_correct_count,
        "over_answer_wrong_count": over_answer_wrong_count,
        "effective_answer_accuracy_when_teacher_answer": safe_divide(
            sum(float(detail.get("answer_score", 0.0)) for detail in teacher_answer_details),
            len(teacher_answer_details),
        ),
    }


def grouped_metrics(details: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        if field_name == "clean_vs_degraded":
            key = clean_vs_degraded_bucket(detail)
        else:
            key = str(detail.get(field_name, "unknown"))
        grouped[key].append(detail)

    summary: dict[str, Any] = {}
    for key, group_details in sorted(grouped.items()):
        metrics = compute_metrics(group_details)
        summary[key] = {
            "total_samples": metrics["total_samples"],
            "teacher_answer_count": metrics["teacher_answer_count"],
            "teacher_abstain_count": metrics["teacher_abstain_count"],
            "judge_success_count": metrics["judge_success_count"],
            "judge_failed_count": metrics["judge_failed_count"],
            "judge_failed_rate": metrics["judge_failed_rate"],
            "inferred_answer_count": metrics["inferred_answer_count"],
            "inferred_abstain_count": metrics["inferred_abstain_count"],
            "teacher_decision_match_rate": metrics["teacher_decision_match_rate"],
            "precise_answer_rate": metrics["precise_answer_rate"],
            "precise_abstain_rate": metrics["precise_abstain_rate"],
            "over_answer_rate": metrics["over_answer_rate"],
            "over_answer_content_accuracy": metrics["over_answer_content_accuracy"],
            "over_answer_correct_count": metrics["over_answer_correct_count"],
            "over_answer_partially_correct_count": metrics["over_answer_partially_correct_count"],
            "over_answer_wrong_count": metrics["over_answer_wrong_count"],
            "over_abstain_rate": metrics["over_abstain_rate"],
            "answer_content_accuracy_when_both_answer": metrics["answer_content_accuracy_when_both_answer"],
            "effective_answer_accuracy_when_teacher_answer": metrics[
                "effective_answer_accuracy_when_teacher_answer"
            ],
        }
    return summary


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
            if clean_vs_degraded_bucket(detail) == "clean" and detail.get("teacher_decision") == "answer"
        ]
        degraded_candidates = [
            detail
            for detail in group_details
            if clean_vs_degraded_bucket(detail) == "degraded" and detail.get("teacher_decision") == "abstain"
        ]
        for clean_detail in clean_candidates:
            for degraded_detail in degraded_candidates:
                eligible_pair_count += 1
                if (
                    clean_detail.get("inferred_decision") == "answer"
                    and degraded_detail.get("inferred_decision") == "abstain"
                ):
                    paired_boundary_success_count += 1

    return {
        "paired_boundary_eligible_count": eligible_pair_count,
        "paired_boundary_success_count": paired_boundary_success_count,
        "paired_boundary_success_rate": safe_divide(paired_boundary_success_count, eligible_pair_count),
    }


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    output_path = resolve_path(args.output_jsonl)
    summary_path = resolve_path(args.summary_json)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    records = select_records(read_jsonl(predictions_path), args.limit)
    selected_keys = {build_done_key(str(record.get("sample_id")), args.model_label) for record in records}
    resume = str_to_bool(args.resume)

    if output_path.exists() and not resume:
        output_path.unlink()

    existing_record_map = load_existing_record_map(output_path, args.model_label) if resume else {}
    done_keys = set(existing_record_map)
    client: SiliconFlowJudgeClient | None = None

    try:
        for record in tqdm(records, desc="Judging unstructured teacher-labeled outputs"):
            sample_id = str(record.get("sample_id"))
            done_key = build_done_key(sample_id, args.model_label)
            if resume and done_key in done_keys:
                continue

            truncated_raw_output = truncate_text(record.get("raw_output"), args.max_raw_output_chars)
            current_teacher_decision = teacher_decision(record)
            gold_answer, gold_source = gold_answer_and_source_for_record(record)

            if current_teacher_decision is None:
                detail = build_detail_record(
                    record,
                    model_label=args.model_label,
                    judge_model="rule",
                    base_raw_output=truncated_raw_output,
                    inferred_decision="unclear",
                    decision_judgement="unclear",
                    answer_judgement="not_applicable",
                    answer_score=0.0,
                    reason="Missing valid teacher decision.",
                    raw_judge_output=None,
                    judge_parse_error=None,
                    judge_success=True,
                    error_type="missing_teacher_decision",
                )
            elif not normalize_text(truncated_raw_output):
                detail = build_detail_record(
                    record,
                    model_label=args.model_label,
                    judge_model="rule",
                    base_raw_output=truncated_raw_output,
                    inferred_decision="unclear",
                    decision_judgement="unclear",
                    answer_judgement="not_applicable",
                    answer_score=0.0,
                    reason="Empty raw_output.",
                    raw_judge_output=None,
                    judge_parse_error=None,
                    judge_success=True,
                    error_type="empty_raw_output",
                )
            else:
                if client is None:
                    client = SiliconFlowJudgeClient(
                        model=args.model,
                        max_retries=args.max_retries,
                        sleep_seconds=args.sleep_seconds,
                    )
                try:
                    judgement, raw_judge_output = client.judge(
                        build_judge_messages(
                            record,
                            raw_output=truncated_raw_output,
                            gold_answer=gold_answer,
                            gold_source=gold_source,
                        ),
                        current_teacher_decision,
                    )
                    detail = build_detail_record(
                        record,
                        model_label=args.model_label,
                        judge_model=args.model,
                        base_raw_output=truncated_raw_output,
                        inferred_decision=judgement["inferred_decision"],
                        decision_judgement=judgement["decision_judgement"],
                        answer_judgement=judgement["answer_judgement"],
                        answer_score=judgement["answer_score"],
                        reason=judgement["reason"],
                        raw_judge_output=raw_judge_output,
                        judge_parse_error=None,
                        judge_success=True,
                        error_type=None,
                    )
                except JudgeRequestFailedError as exc:
                    detail = build_detail_record(
                        record,
                        model_label=args.model_label,
                        judge_model=args.model,
                        base_raw_output=truncated_raw_output,
                        inferred_decision="unclear",
                        decision_judgement="unclear",
                        answer_judgement="not_applicable",
                        answer_score=0.0,
                        reason="Judge failed after retries.",
                        raw_judge_output=exc.raw_output or None,
                        judge_parse_error=exc.parse_error,
                        judge_success=False,
                        error_type="judge_request_failed",
                    )
                except Exception as exc:  # noqa: BLE001
                    detail = build_detail_record(
                        record,
                        model_label=args.model_label,
                        judge_model=args.model,
                        base_raw_output=truncated_raw_output,
                        inferred_decision="unclear",
                        decision_judgement="unclear",
                        answer_judgement="not_applicable",
                        answer_score=0.0,
                        reason="Judge failed unexpectedly.",
                        raw_judge_output=None,
                        judge_parse_error=build_exception_failure_payload(exc),
                        judge_success=False,
                        error_type="judge_unexpected_failed",
                    )

            append_jsonl(output_path, detail)
            existing_record_map[done_key] = detail
            done_keys.add(done_key)
            time.sleep(max(0.0, args.sleep_seconds))
    finally:
        if client is not None:
            client.close()

    details = [existing_record_map[key] for key in sorted(selected_keys) if key in existing_record_map]
    summary: dict[str, Any] = {
        "model_label": args.model_label,
        "predictions_jsonl": str(predictions_path),
        **compute_metrics(details),
        "clean_vs_degraded": grouped_metrics(details, "clean_vs_degraded"),
        "degradation_type_stats": grouped_metrics(details, "degradation_type"),
        "severity_stats": grouped_metrics(details, "severity"),
        **compute_paired_boundary_metrics(details),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
