from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import append_jsonl, ensure_dir, read_jsonl, write_jsonl
from src.json_utils import build_exception_failure_payload
from src.prompt_templates import build_teacher_messages, make_file_image_url
from src.siliconflow_client import TEACHER_MODEL, SiliconFlowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label paired clean/degraded eval JSONL with SiliconFlow teacher_output.")
    parser.add_argument("--input-jsonl", type=str, default="data/interim/test_paired_clean_degraded_index.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="data/processed/test_paired_teacher_labeled.jsonl")
    parser.add_argument("--error-jsonl", type=str, default="outputs/logs/teacher_label_eval_errors.jsonl")
    parser.add_argument("--model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_request_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "clean_sample_id": record.get("clean_sample_id", record.get("sample_id")),
        "image_path": record.get("image_path"),
        "question": record.get("question"),
        "degradation_type": record.get("degradation_type", "none"),
        "severity": record.get("severity", "none"),
        "is_counterfactual": bool(record.get("is_counterfactual", False)),
    }


def load_existing_sample_ids(output_jsonl: Path, *, resume: bool, overwrite: bool) -> set[str]:
    if overwrite:
        write_jsonl(output_jsonl, [])
        return set()
    if not output_jsonl.exists():
        return set()
    if not resume:
        raise FileExistsError(f"Output JSONL exists; pass --overwrite or --resume: {output_jsonl}")
    return {str(record.get("sample_id")) for record in read_jsonl(output_jsonl) if record.get("sample_id")}


def log_failure(error_jsonl: Path, record: dict[str, Any], error_payload: dict[str, Any]) -> None:
    payload = {
        "event": "teacher_label_eval_failed",
        **build_request_context(record),
        "error_payload": error_payload,
    }
    append_jsonl(error_jsonl, payload)


def main() -> None:
    args = parse_args()
    input_jsonl = resolve_path(args.input_jsonl)
    output_jsonl = resolve_path(args.output_jsonl)
    error_jsonl = resolve_path(args.error_jsonl)
    ensure_dir(output_jsonl.parent)
    ensure_dir(error_jsonl.parent)

    existing_ids = load_existing_sample_ids(output_jsonl, resume=args.resume, overwrite=args.overwrite)
    records = read_jsonl(input_jsonl)
    if args.limit is not None:
        records = records[: args.limit]

    client_kwargs: dict[str, Any] = {
        "model": args.model,
        "error_log_path": error_jsonl,
    }
    if args.max_retries is not None:
        client_kwargs["max_retries"] = args.max_retries
    client = SiliconFlowClient(**client_kwargs)

    success_count = 0
    skipped_count = 0
    failure_count = 0
    try:
        for record in tqdm(records, desc="Teacher labeling eval"):
            sample_id = str(record.get("sample_id"))
            if sample_id in existing_ids:
                skipped_count += 1
                continue
            try:
                image_path = resolve_path(str(record["image_path"]))
                image_url = make_file_image_url(image_path)
                messages = build_teacher_messages(record, image_url)
                teacher_output = client.complete_json(
                    messages,
                    temperature=0.2,
                    max_tokens=800,
                    request_context=build_request_context(record),
                    extra_payload={"stream": False, "enable_thinking": False},
                )
                append_jsonl(output_jsonl, {**record, "teacher_output": teacher_output})
                existing_ids.add(sample_id)
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                error_payload = build_exception_failure_payload(exc)
                log_failure(error_jsonl, record, error_payload)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        client.close()

    stats = {
        "input_count": len(records),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "output_jsonl": str(output_jsonl),
        "error_jsonl": str(error_jsonl),
        "model": args.model,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
