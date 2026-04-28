from __future__ import annotations

import argparse
import json
import os
import re
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
ALLOWED_JUDGEMENTS = {"correct", "partially_correct", "wrong"}
JUDGEMENT_SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "wrong": 0.0,
}

NEGATIVE_YESNO_PATTERNS = [
    r"\bno\b",
    r"\bfalse\b",
    r"\babsent\b",
    r"\bwithout\b",
    r"\bnone\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bnot present\b",
    r"\bnot visible\b",
    r"\bnot seen\b",
    r"\bnot show\b",
    r"\bnot shows\b",
    r"\bnot demonstrate\b",
    r"\bnot demonstrates\b",
    r"\bnot reveal\b",
    r"\bnot reveals\b",
    r"\bnot indicate\b",
    r"\bnot indicates\b",
    r"\bnot suggest\b",
    r"\bnot suggests\b",
    r"\bthere is no\b",
    r"\bthere are no\b",
    r"\bno evidence(?: of)?\b",
    r"\bcannot determine\b",
    r"\bunable to determine\b",
    r"\binsufficient evidence\b",
    r"\bnegative for\b",
    r"\bfree of\b",
]
POSITIVE_YESNO_PATTERNS = [
    r"\byes\b",
    r"\btrue\b",
    r"\bpresent\b",
    r"\bvisible\b",
    r"\bseen\b",
    r"\bthere is\b",
    r"\bthere are\b",
    r"\bevidence of\b",
    r"\bpositive for\b",
    r"\bshows?\b",
    r"\bdemonstrates?\b",
    r"\breveals?\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate answer-content correctness on teacher-labeled paired eval predictions."
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="outputs/eval_models_teacher_labeled/sft_lora_vllm/predictions.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/eval_models_teacher_labeled/sft_lora_vllm/answer_correctness_details.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/eval_models_teacher_labeled/sft_lora_vllm/answer_correctness_summary.json",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--model-label", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", type=str, default="true")
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
    return " ".join(str(text or "").strip().lower().split())


def normalize_yes_no(text: Any) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    compact = re.sub(r"[^\w\s]", " ", normalized)
    compact = " ".join(compact.split())
    for pattern in NEGATIVE_YESNO_PATTERNS:
        if re.search(pattern, compact):
            return "no"
    for pattern in POSITIVE_YESNO_PATTERNS:
        if re.search(pattern, compact):
            return "yes"
    return None


def strict_gold_yes_no(text: Any) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    compact = re.sub(r"[^\w\s,\.]", " ", normalized)
    compact = " ".join(compact.split())

    no_patterns = [
        r"^no(?:[\.,]\s*.*)?$",
        r"^false(?:[\.,]\s*.*)?$",
        r"^absent(?:[\.,]\s*.*)?$",
    ]
    yes_patterns = [
        r"^yes(?:[\.,]\s*.*)?$",
        r"^true(?:[\.,]\s*.*)?$",
        r"^present(?:[\.,]\s*.*)?$",
    ]

    for pattern in no_patterns:
        if re.match(pattern, compact):
            return "no"
    for pattern in yes_patterns:
        if re.match(pattern, compact):
            return "yes"
    return None


def build_done_key(sample_id: str, model_label: str) -> str:
    return f"{sample_id}::{model_label}"


def is_teacher_answer_record(record: dict[str, Any]) -> bool:
    teacher_output = record.get("teacher_output")
    return isinstance(teacher_output, dict) and teacher_output.get("decision") == "answer"


def select_teacher_answer_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if not is_teacher_answer_record(record):
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


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


def gold_answer_and_source_for_record(record: dict[str, Any]) -> tuple[str, str]:
    reference_answer = str(record.get("reference_answer") or "").strip()
    if reference_answer:
        return reference_answer, "reference_answer"
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    teacher_answer = str(teacher_output.get("answer") or "").strip()
    if teacher_answer:
        return teacher_answer, "teacher_output.answer"
    return "", "teacher_output.answer"


def answer_type_for_gold(gold_answer: Any) -> str:
    return "yesno" if strict_gold_yes_no(gold_answer) is not None else "open"

def clean_vs_degraded_bucket(detail: dict[str, Any]) -> str:
    if bool(detail.get("is_counterfactual", False)):
        return "degraded"
    if str(detail.get("degradation_type", "none")) != "none":
        return "degraded"
    return "clean"


def build_judge_messages(record: dict[str, Any], gold_answer: str, student_answer: str) -> list[dict[str, Any]]:
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    teacher_explanation = str(teacher_output.get("explanation") or "").strip()
    prompt_lines = [
        "You are an expert medical VQA answer judge. Return strict JSON only.",
        "Judge whether student_answer is semantically correct relative to gold_answer for the given question.",
        "Do not require exact wording.",
        "Synonyms, medical abbreviations, and equivalent expressions can be correct.",
        "A more specific answer that does not contradict gold_answer can be correct.",
        "If laterality, direction, organ, location, or lesion/entity conflicts, it must be wrong.",
        'If the answer is directionally right but too generic or missing a key qualifier, use "partially_correct".',
        '"CT" versus "CT with contrast" is partially_correct.',
        '"CXR", "chest x-ray", and "plain film x-ray" can be correct.',
        '"SVC" and "superior vena cava" can be correct.',
        "Judge only from question, gold_answer, student_answer, optional reference_answer, and optional teacher_explanation.",
        "Do not infer anything from images.",
        "Return exactly this JSON shape:",
        '{"judgement":"correct|partially_correct|wrong","score":1.0,"reason":"one concise sentence"}',
        f"question: {json.dumps(str(record.get('question') or ''), ensure_ascii=False)}",
        f"gold_answer: {json.dumps(gold_answer, ensure_ascii=False)}",
        f"student_answer: {json.dumps(student_answer, ensure_ascii=False)}",
    ]
    reference_answer = str(record.get("reference_answer") or "").strip()
    if reference_answer:
        prompt_lines.append(f"reference_answer: {json.dumps(reference_answer, ensure_ascii=False)}")
    if teacher_explanation:
        prompt_lines.append(f"teacher_explanation: {json.dumps(teacher_explanation, ensure_ascii=False)}")
    prompt = "\n".join(prompt_lines)
    return [
        {"role": "system", "content": "You are a strict medical VQA evaluation judge. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]


def parse_judge_json(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")
    judgement = str(payload.get("judgement", "")).strip()
    if judgement not in ALLOWED_JUDGEMENTS:
        raise ValueError(f"Invalid judgement: {judgement}")
    score = float(payload.get("score", JUDGEMENT_SCORES[judgement]))
    expected_score = JUDGEMENT_SCORES[judgement]
    if score != expected_score:
        score = expected_score
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("Judge reason must be non-empty.")
    return {
        "judgement": judgement,
        "score": score,
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

    def judge(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
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
                    "max_tokens": 256,
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
                return parse_judge_json(last_raw_text), last_raw_text
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


def build_base_detail(
    record: dict[str, Any],
    *,
    model_label: str,
    gold_answer: str,
    gold_source: str,
    answer_type: str,
    judge_model: str,
    student_decision: str,
    student_answer: str,
) -> dict[str, Any]:
    teacher_output = record.get("teacher_output") if isinstance(record.get("teacher_output"), dict) else {}
    teacher_answer = str(teacher_output.get("answer") or "").strip() or None
    return {
        "sample_id": record.get("sample_id"),
        "clean_sample_id": clean_sample_id(record),
        "model_label": model_label,
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "gold_answer": gold_answer,
        "gold_source": gold_source,
        "teacher_answer": teacher_answer,
        "teacher_decision": "answer",
        "student_decision": student_decision,
        "student_answer": student_answer or None,
        "json_parse_success": bool(record.get("json_parse_success", False)),
        "degradation_type": record.get("degradation_type", "none"),
        "severity": record.get("severity", "none"),
        "is_counterfactual": bool(record.get("is_counterfactual", False)),
        "answer_type": answer_type,
        "outcome": None,
        "content_score": 0.0,
        "effective_correct": False,
        "judge_model": judge_model,
        "judge_reason": None,
        "raw_judge_output": None,
        "judge_parse_error": None,
    }


def evaluate_yesno_answer(gold_answer: str, student_answer: str) -> tuple[str, float, str]:
    gold_label = strict_gold_yes_no(gold_answer)
    student_label = normalize_yes_no(student_answer)
    if gold_label is None:
        return "wrong", 0.0, "Gold answer is not a strict yes/no label."
    if student_label is not None and student_label == gold_label:
        return "correct", 1.0, f"Rule-based yes/no match: {student_label}."
    return "wrong", 0.0, "Rule-based yes/no mismatch."


def summarize_details(
    details: list[dict[str, Any]],
    *,
    model_label: str,
    predictions_path: Path,
) -> dict[str, Any]:
    total = len(details)
    valid_student_answer_count = sum(1 for detail in details if detail.get("student_decision") == "answer")
    invalid_output_count = sum(1 for detail in details if detail.get("outcome") == "invalid_output")
    over_abstain_count = sum(1 for detail in details if detail.get("outcome") == "over_abstain")
    judged_outcomes = {"correct", "partially_correct", "wrong"}
    judged_answer_count = sum(1 for detail in details if detail.get("outcome") in judged_outcomes)
    yesno_count = sum(1 for detail in details if detail.get("answer_type") == "yesno")
    yesno_correct_count = sum(
        1
        for detail in details
        if detail.get("answer_type") == "yesno" and detail.get("outcome") == "correct"
    )
    open_count = sum(1 for detail in details if detail.get("answer_type") == "open")
    open_judged = [
        detail for detail in details if detail.get("answer_type") == "open" and detail.get("outcome") in judged_outcomes
    ]
    open_score_sum = sum(float(detail.get("content_score", 0.0)) for detail in open_judged)
    both_answer_judged = [
        detail for detail in details if detail.get("student_decision") == "answer" and detail.get("outcome") in judged_outcomes
    ]

    return {
        "model_label": model_label,
        "predictions_jsonl": str(predictions_path),
        "total_teacher_answer_samples": total,
        "reference_gold_count": sum(1 for detail in details if detail.get("gold_source") == "reference_answer"),
        "teacher_gold_fallback_count": sum(
            1 for detail in details if detail.get("gold_source") == "teacher_output.answer"
        ),
        "valid_student_answer_count": valid_student_answer_count,
        "invalid_output_count": invalid_output_count,
        "over_abstain_count": over_abstain_count,
        "judge_failed_count": sum(1 for detail in details if detail.get("outcome") == "judge_failed"),
        "judged_answer_count": judged_answer_count,
        "yesno_count": yesno_count,
        "yesno_correct_count": yesno_correct_count,
        "yesno_accuracy": safe_divide(yesno_correct_count, yesno_count),
        "open_count": open_count,
        "open_judged_count": len(open_judged),
        "open_score_sum": open_score_sum,
        "open_average_score": safe_divide(open_score_sum, len(open_judged)),
        "correct_count": sum(1 for detail in details if detail.get("outcome") == "correct"),
        "partially_correct_count": sum(1 for detail in details if detail.get("outcome") == "partially_correct"),
        "wrong_count": sum(1 for detail in details if detail.get("outcome") == "wrong"),
        "answer_content_accuracy_when_both_answer": safe_divide(
            sum(float(detail.get("content_score", 0.0)) for detail in both_answer_judged),
            len(both_answer_judged),
        ),
        "effective_answer_accuracy_when_teacher_answer": safe_divide(
            sum(float(detail.get("content_score", 0.0)) for detail in details),
            total,
        ),
        "over_abstain_rate_when_teacher_answer": safe_divide(over_abstain_count, total),
        "invalid_output_rate_when_teacher_answer": safe_divide(invalid_output_count, total),
    }


def grouped_summary(details: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        if field_name == "clean_vs_degraded":
            key = clean_vs_degraded_bucket(detail)
        else:
            key = str(detail.get(field_name, "unknown"))
        grouped[key].append(detail)

    summary: dict[str, Any] = {}
    for key, group_details in sorted(grouped.items()):
        group_summary = summarize_details(
            group_details,
            model_label=str(group_details[0].get("model_label", "")) if group_details else "",
            predictions_path=Path(""),
        )
        summary[key] = {
            "total_teacher_answer_samples": group_summary["total_teacher_answer_samples"],
            "valid_student_answer_count": group_summary["valid_student_answer_count"],
            "over_abstain_count": group_summary["over_abstain_count"],
            "invalid_output_count": group_summary["invalid_output_count"],
            "answer_content_accuracy_when_both_answer": group_summary["answer_content_accuracy_when_both_answer"],
            "effective_answer_accuracy_when_teacher_answer": group_summary[
                "effective_answer_accuracy_when_teacher_answer"
            ],
        }
    return summary


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    output_path = resolve_path(args.output_jsonl)
    summary_path = resolve_path(args.summary_json)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    records = read_jsonl(predictions_path)
    selected_records = select_teacher_answer_records(records, args.limit)
    selected_keys = {build_done_key(str(record.get("sample_id")), args.model_label) for record in selected_records}
    resume = str_to_bool(args.resume)

    if output_path.exists() and not resume:
        output_path.unlink()

    existing_record_map = load_existing_record_map(output_path, args.model_label) if resume else {}
    done_keys = set(existing_record_map)
    client: SiliconFlowJudgeClient | None = None

    try:
        for record in tqdm(selected_records, desc="Evaluating answer correctness"):
            sample_id = str(record.get("sample_id"))
            done_key = build_done_key(sample_id, args.model_label)
            if resume and done_key in done_keys:
                continue

            prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
            student_decision = str(prediction.get("decision", "parse_failed"))
            student_answer = str(prediction.get("answer") or "").strip()
            gold_answer, gold_source = gold_answer_and_source_for_record(record)
            answer_type = answer_type_for_gold(gold_answer)
            detail = build_base_detail(
                record,
                model_label=args.model_label,
                gold_answer=gold_answer,
                gold_source=gold_source,
                answer_type=answer_type,
                judge_model="rule_yesno" if answer_type == "yesno" else args.model,
                student_decision=student_decision if record.get("json_parse_success") else "parse_failed",
                student_answer=student_answer,
            )

            if not record.get("json_parse_success", False):
                detail["outcome"] = "invalid_output"
                detail["judge_reason"] = "Student JSON output could not be parsed."
            elif student_decision != "answer":
                detail["outcome"] = "over_abstain"
                detail["judge_reason"] = "Student did not choose answer when teacher chose answer."
            elif answer_type == "yesno":
                outcome, content_score, reason = evaluate_yesno_answer(gold_answer, student_answer)
                detail["outcome"] = outcome
                detail["content_score"] = content_score
                detail["effective_correct"] = content_score > 0.0
                detail["judge_reason"] = reason
            else:
                detail["judge_model"] = args.model
                if not normalize_text(gold_answer):
                    detail["outcome"] = "judge_failed"
                    detail["judge_reason"] = "Missing gold answer for open-ended judgement."
                    detail["judge_parse_error"] = build_exception_failure_payload(ValueError("Missing gold answer"))
                else:
                    if client is None:
                        client = SiliconFlowJudgeClient(
                            model=args.model,
                            max_retries=args.max_retries,
                            sleep_seconds=args.sleep_seconds,
                        )
                    try:
                        judgement, raw_judge_output = client.judge(
                            build_judge_messages(record, gold_answer=gold_answer, student_answer=student_answer)
                        )
                        detail["outcome"] = judgement["judgement"]
                        detail["content_score"] = judgement["score"]
                        detail["effective_correct"] = judgement["score"] > 0.0
                        detail["judge_reason"] = judgement["reason"]
                        detail["raw_judge_output"] = raw_judge_output
                    except JudgeRequestFailedError as exc:
                        detail["outcome"] = "judge_failed"
                        detail["judge_reason"] = "Judge failed after retries."
                        detail["raw_judge_output"] = exc.raw_output or None
                        detail["judge_parse_error"] = exc.parse_error
                    except Exception as exc:  # noqa: BLE001
                        detail["outcome"] = "judge_failed"
                        detail["judge_reason"] = "Judge failed unexpectedly."
                        detail["judge_parse_error"] = build_exception_failure_payload(exc)

            append_jsonl(output_path, detail)
            existing_record_map[done_key] = detail
            done_keys.add(done_key)
            time.sleep(max(0.0, args.sleep_seconds))
    finally:
        if client is not None:
            client.close()

    details = [existing_record_map[key] for key in sorted(selected_keys) if key in existing_record_map]
    summary: dict[str, Any] = {
        **summarize_details(details, model_label=args.model_label, predictions_path=predictions_path),
        "clean_vs_degraded": grouped_summary(details, "clean_vs_degraded"),
        "degradation_type_stats": grouped_summary(details, "degradation_type"),
        "severity_stats": grouped_summary(details, "severity"),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
