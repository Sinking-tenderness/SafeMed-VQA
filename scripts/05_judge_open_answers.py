from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
ALLOWED_JUDGEMENTS = {"correct", "partially_correct", "wrong", "reasonable_abstain", "over_abstain", "unclear"}
JUDGEMENT_SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "wrong": 0.0,
    "reasonable_abstain": 1.0,
    "over_abstain": 0.0,
    "unclear": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge open-ended VQA answers with a SiliconFlow LLM judge.")
    parser.add_argument("--input-mode", choices=["structured", "unstructured"], default="structured")
    parser.add_argument("--model-label", type=str, default="")
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="outputs/eval_models/sft_lora_vllm/predictions.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/eval_models/sft_lora_vllm/open_answer_judgements.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/eval_models/sft_lora_vllm/open_answer_judge_summary.json",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", type=str, default="true")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def is_yes_no_reference(reference_answer: Any) -> bool:
    return normalize_text(reference_answer) in {"yes", "no"}


def load_done_sample_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done_keys: set[str] = set()
    for record in read_jsonl(output_jsonl):
        sample_id = record.get("sample_id")
        if sample_id is None:
            continue
        model_label = str(record.get("model_label", ""))
        input_mode = str(record.get("input_mode", "structured"))
        done_keys.add(build_done_key(str(sample_id), model_label, input_mode))
    return done_keys


def build_done_key(sample_id: str, model_label: str, input_mode: str) -> str:
    return f"{sample_id}::{model_label}::{input_mode}"


def select_open_samples(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        if is_yes_no_reference(record.get("reference_answer")):
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def model_fields(record: dict[str, Any], input_mode: str) -> tuple[str, str]:
    raw_output = str(record.get("raw_output") or "")
    if input_mode == "unstructured":
        return "unstructured", raw_output[:1500]
    if not record.get("json_parse_success", False):
        return "parse_failed", raw_output[:1000]
    prediction = record.get("prediction") or {}
    return str(prediction.get("decision", "")), str(prediction.get("answer", ""))


def build_judge_messages(record: dict[str, Any], model_decision: str, model_answer: str) -> list[dict[str, Any]]:
    prompt = (
        "You are an expert medical VQA answer judge. Return strict JSON only.\n"
        "Judge whether model_answer is semantically consistent with reference_answer in the question context.\n"
        "Also infer whether the model is answering, abstaining, or unclear from the text itself.\n"
        "Do not require exact match. Synonyms, abbreviations, equivalent medical expressions, and more specific "
        "non-contradictory answers can be correct.\n"
        '"chest x-ray", "CXR", and "plain film x-ray" are equivalent. "SVC" and "superior vena cava" are equivalent. '
        '"small bowel", "small intestines", and "bowel loops" can be equivalent or partially equivalent when context fits.\n'
        '"CT" vs "CT with contrast" is partially correct, not fully correct.\n'
        "If laterality, organ, entity, or lesion location conflicts, judge wrong.\n"
        "For unstructured outputs, do not punish the absence of a JSON schema; judge only the semantic content.\n"
        "Judge only from question, reference_answer, model_decision, and model_answer. Do not infer from images.\n"
        "Inferred decision rules:\n"
        "- answer: model_answer gives a clear answer or obvious leaning.\n"
        "- abstain: model_answer clearly says it cannot determine, evidence is insufficient, or it cannot answer.\n"
        "- unclear: neither a clear answer nor a clear abstention.\n"
        "Judgement rules:\n"
        "- correct: semantically equivalent, including acceptable synonyms or more specific non-contradictory answers.\n"
        "- partially_correct: directionally right but incomplete, too generic, or missing a key qualifier.\n"
        "- wrong: conflicts with reference, wrong site/direction/entity, or does not answer the question.\n"
        "- reasonable_abstain: model abstains and the reason is reasonable, such as insufficient image evidence, missing "
        "sequence/contrast phase, clinical context required, or the question should not be answered from image alone.\n"
        "- over_abstain: reference_answer is answerable but model unnecessarily refuses.\n"
        "- unclear: model_answer does not contain a clear answer or clear abstention, or cannot be judged.\n"
        "- parse_failed inputs are usually wrong unless model_answer clearly contains the correct answer.\n"
        "Scores must be: correct=1.0, partially_correct=0.5, wrong=0.0, reasonable_abstain=1.0, over_abstain=0.0, unclear=0.0.\n"
        "Return exactly this JSON shape:\n"
        '{"inferred_decision":"answer|abstain|unclear","judgement":"correct|partially_correct|wrong|reasonable_abstain|over_abstain|unclear","score":1.0,"reason":"one concise sentence"}\n'
        f"question: {record.get('question')}\n"
        f"reference_answer: {record.get('reference_answer')}\n"
        f"model_decision: {model_decision}\n"
        f"model_answer: {model_answer}"
    )
    return [
        {"role": "system", "content": "You are a strict medical VQA evaluation judge. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]


def parse_judge_json(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")
    inferred_decision = str(payload.get("inferred_decision", "")).strip()
    if inferred_decision not in ALLOWED_INFERRED_DECISIONS:
        raise ValueError(f"Invalid inferred_decision: {inferred_decision}")
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
        "inferred_decision": inferred_decision,
        "judgement": judgement,
        "score": score,
        "reason": reason,
    }


class SiliconFlowJudgeClient:
    def __init__(self, model: str, max_retries: int) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY in environment or .env")
        self.model = model
        self.max_retries = max_retries
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

    def judge(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
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
                try:
                    return parse_judge_json(last_raw_text), last_raw_text, None
                except Exception as parse_exc:  # noqa: BLE001
                    last_error = parse_exc
                    last_parse_error = build_exception_failure_payload(parse_exc)
                    if attempt >= self.max_retries:
                        break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_parse_error = build_exception_failure_payload(exc)
                if attempt >= self.max_retries:
                    break
            time.sleep(min(2 ** (attempt - 1), 30))
        raise RuntimeError(f"Judge request failed after retries: {last_error}; raw={last_raw_text[:500]}") from last_error


def build_output_record(
    record: dict[str, Any],
    judgement: dict[str, Any],
    model_decision: str,
    model_answer: str,
    judge_model: str,
    *,
    model_label: str,
    input_mode: str,
) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "model_label": model_label,
        "input_mode": input_mode,
        "model_decision": model_decision,
        "model_answer": model_answer,
        "inferred_decision": judgement["inferred_decision"],
        "judgement": judgement["judgement"],
        "score": judgement["score"],
        "reason": judgement["reason"],
        "judge_model": judge_model,
    }


def summarize(
    judgements: list[dict[str, Any]],
    total_open_samples: int,
    parse_failed_count: int,
    *,
    input_mode: str,
    model_label: str,
) -> dict[str, Any]:
    judged_count = len(judgements)
    distribution = Counter(record["judgement"] for record in judgements)
    inferred_distribution = Counter(record.get("inferred_decision", "unclear") for record in judgements)
    average_score = sum(float(record["score"]) for record in judgements) / judged_count if judged_count else 0.0
    answered = [record for record in judgements if record.get("inferred_decision") == "answer"]
    abstained = [record for record in judgements if record.get("inferred_decision") == "abstain"]
    unclear = [record for record in judgements if record.get("inferred_decision") == "unclear"]
    answered_distribution = Counter(record["judgement"] for record in answered)
    abstained_distribution = Counter(record["judgement"] for record in abstained)
    correct_count = answered_distribution["correct"]
    partial_count = answered_distribution["partially_correct"]
    reasonable_abstain_count = abstained_distribution["reasonable_abstain"]
    return {
        "total_open_samples": total_open_samples,
        "judged_count": judged_count,
        "judgement_distribution": dict(distribution),
        "inferred_decision_distribution": dict(inferred_distribution),
        "average_score": average_score,
        "input_mode": input_mode,
        "model_label": model_label,
        "answered_open_count": len(answered),
        "abstained_open_count": len(abstained),
        "unclear_open_count": len(unclear),
        "open_answer_accuracy": ((correct_count + 0.5 * partial_count) / len(answered)) if answered else 0.0,
        "abstain_quality": (reasonable_abstain_count / len(abstained)) if abstained else 0.0,
        "open_utility_score": average_score,
        "parse_failed_count": parse_failed_count,
    }


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions_jsonl)
    output_path = resolve_path(args.output_jsonl)
    summary_path = resolve_path(args.summary_json)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)

    predictions = read_jsonl(predictions_path)
    open_samples = select_open_samples(predictions, args.limit)
    resume = str_to_bool(args.resume)
    done_ids = load_done_sample_ids(output_path) if resume else set()
    existing_records = read_jsonl(output_path) if output_path.exists() and resume else []
    parse_failed_count = sum(1 for record in open_samples if not record.get("json_parse_success", False))

    client = SiliconFlowJudgeClient(model=args.model, max_retries=args.max_retries)
    try:
        for record in tqdm(open_samples, desc="Judging open answers"):
            sample_id = str(record.get("sample_id"))
            done_key = build_done_key(sample_id, args.model_label, args.input_mode)
            if resume and done_key in done_ids:
                continue
            model_decision, model_answer = model_fields(record, args.input_mode)
            messages = build_judge_messages(record, model_decision, model_answer)
            try:
                judgement, raw_judge_output, judge_parse_error = client.judge(messages)
                output_record = build_output_record(
                    record,
                    judgement,
                    model_decision,
                    model_answer,
                    args.model,
                    model_label=args.model_label,
                    input_mode=args.input_mode,
                )
                output_record["raw_judge_output"] = raw_judge_output
                if judge_parse_error is not None:
                    output_record["judge_parse_error"] = judge_parse_error
            except Exception as exc:  # noqa: BLE001
                error_payload = build_exception_failure_payload(exc)
                output_record = {
                    "sample_id": record.get("sample_id"),
                    "question": record.get("question"),
                    "reference_answer": record.get("reference_answer"),
                    "model_label": args.model_label,
                    "input_mode": args.input_mode,
                    "model_decision": model_decision,
                    "model_answer": model_answer,
                    "inferred_decision": "unclear",
                    "judgement": "wrong",
                    "score": 0.0,
                    "reason": "Judge failed after retries.",
                    "judge_model": args.model,
                    "judge_parse_error": error_payload,
                }
            append_jsonl(output_path, output_record)
            existing_records.append(output_record)
            done_ids.add(done_key)
            time.sleep(max(0.0, args.sleep_seconds))
    finally:
        client.close()

    filtered_records = [
        record
        for record in existing_records
        if str(record.get("model_label", "")) == args.model_label and str(record.get("input_mode", "structured")) == args.input_mode
    ]
    summary = summarize(
        filtered_records,
        total_open_samples=len(open_samples),
        parse_failed_count=parse_failed_count,
        input_mode=args.input_mode,
        model_label=args.model_label,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
