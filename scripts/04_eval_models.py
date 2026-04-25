from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

    
from src.dataset_utils import ROOT, ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from src.json_utils import build_exception_failure_payload, safe_parse_and_validate_json_output
from src.model_utils import build_inference_messages, load_image, load_processor, load_student_model


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
DEFAULT_SFT_ADAPTER = "checkpoints/sft_qwen3vl8b_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base / SFT-LoRA / DPO-LoRA models on a shared test set.")
    parser.add_argument("--model_mode", choices=["base", "sft_lora", "dpo_lora"], required=True)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-adapter-path", type=str, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--dpo-adapter-path", type=str, default=None)
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--use-flash-attention-2", action="store_true")
    parser.add_argument("--vllm-max-lora-rank", type=int, default=64)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95)
    return parser.parse_args()


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def normalize_answer(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def resolve_adapter_path(args: argparse.Namespace) -> Path | None:
    if args.model_mode == "base":
        return None
    if args.model_mode == "sft_lora":
        adapter_path = resolve_path(args.sft_adapter_path)
        if adapter_path is None or not adapter_path.exists():
            raise FileNotFoundError(f"SFT adapter path not found: {adapter_path}")
        return adapter_path
    adapter_path = resolve_path(args.dpo_adapter_path)
    if adapter_path is None or not adapter_path.exists():
        raise FileNotFoundError(f"DPO adapter path not found: {adapter_path}")
    return adapter_path


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        *,
        base_model_path: Path,
        adapter_path: Path | None,
        torch_dtype: str,
        max_new_tokens: int,
        use_flash_attention_2: bool,
    ) -> None:
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.processor = load_processor(str(base_model_path))
        self.model = load_student_model(
            model_name=str(base_model_path),
            torch_dtype=torch_dtype,
            adapter_path=adapter_path,
            trainable_adapter=False,
            use_flash_attention_2=use_flash_attention_2,
        )
        if hasattr(self.model, "eval"):
            self.model.eval()

    def generate_raw(self, image_path: str | Path, question: str) -> str:
        messages = build_inference_messages(question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = load_image(image_path)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if hasattr(self.model, "device") and getattr(self.model.device, "type", "cpu") != "cpu":
            inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        completion = generated[0][prompt_len:]
        return self.processor.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def close(self) -> None:
        return None


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        *,
        base_model_path: Path,
        adapter_path: Path | None,
        max_new_tokens: int,
        vllm_max_lora_rank: int,
        vllm_max_model_len: int,
        vllm_gpu_memory_utilization: float,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("vLLM is not installed in the current environment.") from exc

        self.processor = load_processor(str(base_model_path))
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        llm_kwargs: dict[str, Any] = {
            "model": str(base_model_path),
            "trust_remote_code": True,
            "max_model_len": vllm_max_model_len,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
        }

        self.lora_request = None
        if adapter_path is not None:
            try:
                from vllm.lora.request import LoRARequest  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("vLLM LoRA support is unavailable in the current environment.") from exc
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = vllm_max_lora_rank
            adapter_name = adapter_path.name
            self.lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=abs(hash((adapter_name, str(adapter_path)))) % (10**9),
                lora_path=str(adapter_path),
            )

        self.llm = LLM(**llm_kwargs)

    def generate_raw(self, image_path: str | Path, question: str) -> str:
        messages = build_inference_messages(question)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_payload: dict[str, Any] = {
            "prompt": prompt,
            "multi_modal_data": {"image": load_image(image_path)},
        }
        generate_kwargs: dict[str, Any] = {}
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request
        outputs = self.llm.generate(prompt_payload, sampling_params=self.sampling_params, **generate_kwargs)
        return outputs[0].outputs[0].text.strip()

    def close(self) -> None:
        return None


def build_backend(
    *,
    backend_name: str,
    base_model_path: Path,
    adapter_path: Path | None,
    test_records: list[dict[str, Any]],
    max_new_tokens: int,
    torch_dtype: str,
    use_flash_attention_2: bool,
    vllm_max_lora_rank: int,
    vllm_max_model_len: int,
    vllm_gpu_memory_utilization: float,
) -> tuple[Any, str]:
    def build_transformers_backend() -> TransformersBackend:
        return TransformersBackend(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            use_flash_attention_2=use_flash_attention_2,
        )

    if backend_name == "transformers":
        return build_transformers_backend(), "transformers"

    def build_and_warm_vllm() -> tuple[VLLMBackend, str]:
        backend = VLLMBackend(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            vllm_max_lora_rank=vllm_max_lora_rank,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        if test_records:
            first = test_records[0]
            backend.generate_raw(first["image_path"], first["question"])
        return backend, "vllm"

    if backend_name == "vllm":
        return build_and_warm_vllm()

    try:
        return build_and_warm_vllm()
    except Exception as exc:  # noqa: BLE001
        print(f"vLLM backend unavailable or incompatible, falling back to transformers: {exc}")
        return build_transformers_backend(), "transformers"


def compute_teacher_field_consistency(
    prediction: dict[str, Any] | None,
    teacher_output: dict[str, Any] | None,
) -> dict[str, bool | None]:
    fields = ["decision", "abstain_type", "risk_level", "answer"]
    if prediction is None or not isinstance(teacher_output, dict):
        return {field: None for field in fields}

    consistency: dict[str, bool | None] = {}
    for field in fields:
        if field not in teacher_output:
            consistency[field] = None
            continue
        if field == "answer":
            consistency[field] = normalize_answer(prediction.get(field)) == normalize_answer(teacher_output.get(field))
        else:
            consistency[field] = prediction.get(field) == teacher_output.get(field)
    return consistency


def summarize_results(
    *,
    model_mode: str,
    backend_requested: str,
    backend_used: str,
    base_model_path: Path,
    adapter_path: Path | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    parsed = [record for record in results if record["json_parse_success"]]
    parsed_count = len(parsed)
    parsed_predictions = [record["prediction"] for record in parsed if isinstance(record.get("prediction"), dict)]

    decision_distribution = Counter(prediction.get("decision", "missing") for prediction in parsed_predictions)
    abstain_distribution = Counter(prediction.get("abstain_type", "missing") for prediction in parsed_predictions)
    risk_distribution = Counter(prediction.get("risk_level", "missing") for prediction in parsed_predictions)

    clean_vs_degraded: dict[str, dict[str, Any]] = {}
    grouped_decisions: dict[str, Counter[str]] = defaultdict(Counter)
    for record in parsed:
        prediction = record["prediction"]
        bucket = "clean" if record.get("degradation_type", "none") == "none" else "degraded"
        grouped_decisions[bucket][prediction.get("decision", "missing")] += 1
    for bucket in ["clean", "degraded"]:
        answer_count = grouped_decisions[bucket]["answer"]
        abstain_count = grouped_decisions[bucket]["abstain"]
        parsed_bucket_total = answer_count + abstain_count
        clean_vs_degraded[bucket] = {
            "parsed_samples": parsed_bucket_total,
            "answer_count": answer_count,
            "abstain_count": abstain_count,
            "answer_ratio": (answer_count / parsed_bucket_total) if parsed_bucket_total else 0.0,
            "abstain_ratio": (abstain_count / parsed_bucket_total) if parsed_bucket_total else 0.0,
        }

    def summarize_group(field_name: str) -> dict[str, Any]:
        group_stats: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in results:
            grouped[str(record.get(field_name, "unknown"))].append(record)
        for group_name, group_records in sorted(grouped.items()):
            parsed_group = [record for record in group_records if record["json_parse_success"]]
            group_decisions = Counter(record["prediction"].get("decision", "missing") for record in parsed_group)
            group_stats[group_name] = {
                "total_samples": len(group_records),
                "parsed_samples": len(parsed_group),
                "decision_distribution": dict(group_decisions),
            }
        return group_stats

    consistency_summary: dict[str, dict[str, Any]] = {}
    for field in ["decision", "abstain_type", "risk_level", "answer"]:
        available = [
            record["teacher_field_consistency"][field]
            for record in results
            if record["teacher_field_consistency"].get(field) is not None
        ]
        matched = sum(1 for value in available if value is True)
        consistency_summary[field] = {
            "available_count": len(available),
            "matched_count": matched,
            "match_rate": (matched / len(available)) if available else None,
        }

    return {
        "model_mode": model_mode,
        "backend_requested": backend_requested,
        "backend_used": backend_used,
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "total_samples": total,
        "json_parse_success_count": parsed_count,
        "json_parse_success_rate": (parsed_count / total) if total else 0.0,
        "decision_distribution": dict(decision_distribution),
        "abstain_type_distribution": dict(abstain_distribution),
        "risk_level_distribution": dict(risk_distribution),
        "clean_vs_degraded": clean_vs_degraded,
        "degradation_type_stats": summarize_group("degradation_type"),
        "severity_stats": summarize_group("severity"),
        "teacher_field_consistency": consistency_summary,
    }


def update_comparison_summary(output_dir: Path, model_mode: str, summary: dict[str, Any], run_dir: Path) -> None:
    comparison_path = output_dir / "model_comparison_summary.json"
    if comparison_path.exists():
        payload = read_json(comparison_path)
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}

    summary_key = f"{model_mode}_{summary['backend_used']}"

    payload[summary_key] = {
        "model_mode": model_mode,
        "summary_path": str((run_dir / "summary.json").resolve()),
        "results_path": str((run_dir / "predictions.jsonl").resolve()),
        "backend_used": summary["backend_used"],
        "json_parse_success_rate": summary["json_parse_success_rate"],
        "teacher_field_consistency": summary["teacher_field_consistency"],
        "decision_distribution": summary["decision_distribution"],
    }
    write_json(comparison_path, payload)


def main() -> None:
    args = parse_args()
    base_model_path = resolve_path(args.base_model_path)
    if base_model_path is None or not base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")

    adapter_path = resolve_adapter_path(args)
    test_jsonl_path = resolve_path(args.test_jsonl)
    if test_jsonl_path is None or not test_jsonl_path.exists():
        raise FileNotFoundError(f"Test JSONL path not found: {test_jsonl_path}")

    output_dir = ensure_dir(resolve_path(args.output_dir))
    test_records = read_jsonl(test_jsonl_path)
    if not test_records:
        raise RuntimeError("Test JSONL is empty; cannot evaluate models.")

    backend, backend_used = build_backend(
        backend_name=args.backend,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        test_records=test_records,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        use_flash_attention_2=args.use_flash_attention_2,
        vllm_max_lora_rank=args.vllm_max_lora_rank,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    run_dir = ensure_dir(output_dir / f"{args.model_mode}_{backend_used}")
    results: list[dict[str, Any]] = []
    try:
        for record in tqdm(test_records, desc=f"Evaluating {args.model_mode} ({backend_used})"):
            raw_text: str | None = None
            prediction: dict[str, Any] | None = None
            parse_error: dict[str, Any] | None = None
            try:
                raw_text = backend.generate_raw(record["image_path"], record["question"])
                prediction, parse_error = safe_parse_and_validate_json_output(raw_text)
            except Exception as exc:  # noqa: BLE001
                parse_error = build_exception_failure_payload(exc)

            teacher_output = record.get("teacher_output")
            result_record = {
                "sample_id": record.get("sample_id"),
                "image_path": record.get("image_path"),
                "question": record.get("question"),
                "reference_answer": record.get("reference_answer", record.get("answer", "")),
                "split": record.get("split"),
                "degradation_type": record.get("degradation_type", "none"),
                "severity": record.get("severity", "none"),
                "is_counterfactual": bool(record.get("is_counterfactual", False)),
                "teacher_output": teacher_output,
                "model_mode": args.model_mode,
                "backend_requested": args.backend,
                "backend_used": backend_used,
                "json_parse_success": prediction is not None,
                "prediction": prediction,
                "raw_output": raw_text,
                "parse_error": parse_error,
            }
            result_record["teacher_field_consistency"] = compute_teacher_field_consistency(prediction, teacher_output)
            results.append(result_record)
    finally:
        backend.close()

    summary = summarize_results(
        model_mode=args.model_mode,
        backend_requested=args.backend,
        backend_used=backend_used,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        results=results,
    )

    write_jsonl(run_dir / "predictions.jsonl", results)
    write_json(run_dir / "summary.json", summary)
    update_comparison_summary(output_dir, args.model_mode, summary, run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {run_dir / 'predictions.jsonl'}")
    print(f"Wrote summary to {run_dir / 'summary.json'}")
    print(f"Updated comparison file at {output_dir / 'model_comparison_summary.json'}")


if __name__ == "__main__":
    main()
