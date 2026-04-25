from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import append_jsonl, ensure_dir, read_jsonl, write_jsonl
from src.image_augment import create_augmented_sample, severity_choices
from src.json_utils import build_exception_failure_payload
from src.prompt_templates import build_analysis_entry, build_teacher_messages, make_file_image_url
from src.siliconflow_client import SiliconFlowClient


LOGGER = logging.getLogger("datagen_pro_plus")
DEGRADATION_TYPES = [
    "gaussian_blur",
    "speckle_noise",
    "resolution_drop",
    "contrast_brightness_shift",
    "random_crop",
    "local_occlusion",
    "center_mask",
    "border_truncate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SafeMed-VQA Pro++ training data with SiliconFlow teacher.")
    parser.add_argument("--config", type=str, default="configs/datagen.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-first-n", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.md5(f"{base_seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_augmented_variant_stub(
    sample: dict[str, Any],
    augmented_root: Path,
    *,
    severity: str,
    degradation_type: str,
    variant_seed: int,
) -> dict[str, Any]:
    suffix = Path(sample["image_path"]).suffix or ".png"
    image_path = augmented_root / f"{sample['sample_id']}_{degradation_type}_{severity}{suffix}"
    return {
        **sample,
        "sample_id": f"{sample['sample_id']}__{degradation_type}_{severity}",
        "image_path": str(image_path.resolve()),
        "is_counterfactual": True,
        "degradation_type": degradation_type,
        "severity": severity,
        "source_sample_id": sample["sample_id"],
        "_source_image_path": sample["image_path"],
        "_variant_seed": variant_seed,
        "_requires_augmentation": True,
    }


def build_sample_variants(
    sample: dict[str, Any],
    augmented_root: Path,
    cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    variants = [
        {
            **sample,
            "is_counterfactual": False,
            "degradation_type": "none",
            "severity": "none",
            "_requires_augmentation": False,
        }
    ]
    aug_cfg = cfg.get("augmentation", {})
    severity_pool = severity_choices(aug_cfg.get("severity_weights", {}))
    unique_severities = list(dict.fromkeys(severity_pool))
    candidate_pairs = [(degradation_type, severity) for degradation_type in DEGRADATION_TYPES for severity in unique_severities]
    rng.shuffle(candidate_pairs)
    per_sample_variants = int(aug_cfg.get("per_sample_variants", 4))
    selected_pairs = candidate_pairs[: min(per_sample_variants, len(candidate_pairs))]

    LOGGER.info(
        "variant_plan sample_id=%s candidate_combo_count=%s selected_combos=%s",
        sample["sample_id"],
        len(candidate_pairs),
        selected_pairs,
    )

    for degradation_type, severity in selected_pairs:
        variants.append(
            build_augmented_variant_stub(
                sample=sample,
                augmented_root=augmented_root,
                severity=severity,
                degradation_type=degradation_type,
                variant_seed=rng.randint(0, 2**31 - 1),
            )
        )
    return variants, 0


def build_request_context(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": variant.get("sample_id"),
        "source_sample_id": variant.get("source_sample_id", variant.get("sample_id")),
        "image_path": variant.get("image_path"),
        "question": variant.get("question"),
        "degradation_type": variant.get("degradation_type", "none"),
        "severity": variant.get("severity", "none"),
    }


def load_existing_records(records_path: Path) -> dict[str, dict[str, Any]]:
    if not records_path.exists():
        return {}
    return {record["sample_id"]: record for record in read_jsonl(records_path)}


def log_cumulative_counts(
    *,
    processed_samples: int,
    total_samples: int,
    success_count: int,
    failure_count: int,
    skipped_count: int,
) -> None:
    LOGGER.info(
        "cumulative_counts processed_samples=%s/%s variants_succeeded=%s failed=%s skipped=%s",
        processed_samples,
        total_samples,
        success_count,
        failure_count,
        skipped_count,
    )


def materialize_variant_if_needed(variant: dict[str, Any], augmented_root: Path) -> dict[str, Any]:
    expected_image_path = str(variant.get("image_path"))
    if not variant.get("_requires_augmentation"):
        if not Path(expected_image_path).exists():
            raise RuntimeError(
                "Base image was not found during variant materialization: "
                f"sample_id={variant.get('sample_id')} "
                f"source_sample_id={variant.get('source_sample_id', variant.get('sample_id'))} "
                f"degradation_type={variant.get('degradation_type', 'none')} "
                f"severity={variant.get('severity', 'none')} "
                f"expected_image_path={expected_image_path}"
            )
        return {key: value for key, value in variant.items() if not key.startswith("_")}
    image_path = Path(variant["image_path"])
    if image_path.exists():
        LOGGER.info(
            "variant_materialized sample_id=%s source_sample_id=%s output_path=%s write_success=%s",
            variant.get("sample_id"),
            variant.get("source_sample_id", variant.get("sample_id")),
            str(image_path),
            True,
        )
        return {key: value for key, value in variant.items() if not key.startswith("_")}
    LOGGER.info(
        "variant_materialization_start sample_id=%s source_sample_id=%s output_path=%s",
        variant.get("sample_id"),
        variant.get("source_sample_id", variant.get("sample_id")),
        expected_image_path,
    )
    materialized = create_augmented_sample(
        sample=variant,
        output_dir=augmented_root,
        severity=variant["severity"],
        degradation_type=variant["degradation_type"],
        rng=random.Random(int(variant["_variant_seed"])),
    )
    materialized_path = Path(materialized["image_path"])
    write_success = materialized_path.exists()
    LOGGER.info(
        "variant_materialization_end sample_id=%s source_sample_id=%s output_path=%s write_success=%s",
        materialized.get("sample_id"),
        materialized.get("source_sample_id", materialized.get("sample_id")),
        materialized.get("image_path"),
        write_success,
    )
    if not write_success:
        raise RuntimeError(
            "Augmented image was not materialized: "
            f"sample_id={materialized.get('sample_id')} "
            f"source_sample_id={materialized.get('source_sample_id', materialized.get('sample_id'))} "
            f"degradation_type={materialized.get('degradation_type', 'none')} "
            f"severity={materialized.get('severity', 'none')} "
            f"expected_image_path={materialized.get('image_path')}"
        )
    return {key: value for key, value in materialized.items() if not key.startswith("_")}


def print_sample_analysis(records: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0 or not records:
        return
    print(f"First {min(limit, len(records))} generated samples:")
    for record in records[:limit]:
        print(json.dumps(build_analysis_entry(record), ensure_ascii=False))


def build_teacher_client(
    teacher_cfg: dict[str, Any],
    error_log_path: Path,
    *,
    enable_error_logging: bool = True,
) -> SiliconFlowClient:
    return SiliconFlowClient(
        model=teacher_cfg.get("model", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
        timeout=int(teacher_cfg.get("timeout", 120)),
        connectivity_timeout=int(teacher_cfg.get("connectivity_check_timeout", 30)),
        max_retries=int(teacher_cfg.get("max_retries", 3)),
        error_log_path=error_log_path,
        enable_error_logging=enable_error_logging,
    )


def build_output_record(variant: dict[str, Any], teacher_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": variant["sample_id"],
        "image_path": variant["image_path"],
        "question": variant["question"],
        "reference_answer": variant.get("answer", ""),
        "split": variant.get("split", "train"),
        "is_counterfactual": bool(variant.get("is_counterfactual")),
        "degradation_type": variant.get("degradation_type", "none"),
        "severity": variant.get("severity", "none"),
        "source_sample_id": variant.get("source_sample_id", variant["sample_id"]),
        "teacher_output": teacher_output,
    }


def build_teacher_request_payload(teacher_cfg: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stream": False,
        "enable_thinking": bool(teacher_cfg.get("enable_thinking", False)),
    }
    if payload["enable_thinking"] and teacher_cfg.get("thinking_budget") is not None:
        payload["thinking_budget"] = int(teacher_cfg["thinking_budget"])
    return payload


def process_variant_worker(
    variant: dict[str, Any],
    augmented_root: Path,
    teacher_cfg: dict[str, Any],
    error_log_path: Path,
) -> dict[str, Any]:
    context = build_request_context(variant)
    try:
        materialized_variant = materialize_variant_if_needed(variant, augmented_root)
        context = build_request_context(materialized_variant)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "stage": "materialization",
            "event": "variant_materialization_failed",
            "variant": variant,
            "context": context,
            "error_payload": build_exception_failure_payload(exc),
        }

    try:
        image_url = make_file_image_url(materialized_variant["image_path"])
        messages = build_teacher_messages(materialized_variant, image_url)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "stage": "preflight",
            "event": "variant_preflight_failed",
            "variant": materialized_variant,
            "context": context,
            "error_payload": build_exception_failure_payload(exc),
        }

    client = build_teacher_client(teacher_cfg, error_log_path, enable_error_logging=False)
    try:
        teacher_output = client.complete_json(
            messages,
            temperature=float(teacher_cfg.get("temperature", 0.2)),
            max_tokens=int(teacher_cfg.get("max_tokens", 800)),
            request_context=context,
            extra_payload=build_teacher_request_payload(teacher_cfg),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "stage": "completion",
            "event": "variant_completion_failed",
            "variant": materialized_variant,
            "context": context,
            "error_payload": build_exception_failure_payload(exc),
        }
    finally:
        client.close()

    return {
        "ok": True,
        "variant": materialized_variant,
        "context": context,
        "teacher_output": teacher_output,
    }


def main() -> None:
    args = parse_args()
    configure_logging()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config.get("seed", 42))
    datagen_cfg = config.get("datagen", {})
    teacher_cfg = config.get("teacher", {})
    logging_cfg = config.get("logging", {})
    output_cfg = config["output"]
    train_index_path = ROOT / output_cfg["train_index"]
    records_path = ROOT / output_cfg["teacher_train_jsonl"]
    status_path = ROOT / output_cfg["datagen_status_json"]
    error_log_path = ROOT / output_cfg["datagen_error_log"]
    augmented_root = ROOT / output_cfg["augmented_root"]
    ensure_dir(augmented_root)
    num_workers = max(1, int(args.num_workers if args.num_workers is not None else datagen_cfg.get("num_workers", 1)))
    client = build_teacher_client(teacher_cfg, error_log_path)
    status_state = client.load_status_state(status_path)
    existing_records_by_id = load_existing_records(records_path)
    for record in existing_records_by_id.values():
        context = build_request_context(record)
        client.update_status_entry(
            status_state,
            status_key=record["sample_id"],
            status="success",
            increment_attempt=False,
            request_context=context,
            output_record_path=str(records_path),
        )

    samples = read_jsonl(train_index_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    test_samples = [sample for sample in samples if str(sample.get("split", "")).lower() == "test"]
    samples = [sample for sample in samples if str(sample.get("split", "")).lower() != "test"]
    for sample in test_samples:
        context = build_request_context(sample)
        client.update_status_entry(
            status_state,
            status_key=sample["sample_id"],
            status="skipped",
            increment_attempt=False,
            request_context=context,
            note="Excluded from distillation because split=test.",
        )
    client.save_status_state(status_path, status_state)
    if not samples:
        if not records_path.exists():
            write_jsonl(records_path, [])
        client.close()
        print(
            f"Teacher data generation summary: samples=0 variants_succeeded=0 failures=0 skipped={len(test_samples)} "
            f"output={records_path}"
        )
        return

    success_count = 0
    failure_count = 0
    skipped_count = len(test_samples)
    progress_log_interval = max(1, int(logging_cfg.get("progress_log_interval", 10)))
    total_samples = len(samples)
    max_in_flight = max(1, num_workers * 2)
    planned_tasks: list[dict[str, Any]] = []
    sample_stats: dict[str, dict[str, Any]] = {}
    completed_samples = 0

    def finalize_base_sample(base_sample_id: str) -> None:
        nonlocal completed_samples
        stats = sample_stats[base_sample_id]
        if stats.get("finalized"):
            return
        stats["finalized"] = True
        completed_samples += 1
        LOGGER.info(
            "base_sample_end sample_id=%s successes=%s failed=%s skipped=%s",
            base_sample_id,
            stats["successes"],
            stats["failed"],
            stats["skipped"],
        )
        if completed_samples % progress_log_interval == 0 or completed_samples == total_samples:
            log_cumulative_counts(
                processed_samples=completed_samples,
                total_samples=total_samples,
                success_count=success_count,
                failure_count=failure_count,
                skipped_count=skipped_count,
            )

    for sample_index, sample in enumerate(tqdm(samples, desc="Planning teacher supervision"), start=1):
        base_sample_id = sample["sample_id"]
        LOGGER.info("base_sample_start sample_id=%s index=%s/%s", base_sample_id, sample_index, total_samples)
        sample_stats[base_sample_id] = {
            "successes": 0,
            "failed": 0,
            "skipped": 0,
            "pending": 0,
            "finalized": False,
        }
        try:
            sample_rng = random.Random(stable_sample_seed(seed, base_sample_id))
            variants, variant_skips = build_sample_variants(sample, augmented_root, config, sample_rng)
            skipped_count += variant_skips
            sample_stats[base_sample_id]["skipped"] += variant_skips
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            sample_stats[base_sample_id]["failed"] += 1
            failure_payload = build_exception_failure_payload(exc)
            client.log_error(
                event="variant_build_failed",
                error=failure_payload["error"],
                request_context=build_request_context(sample),
            )
            client.update_status_entry(
                status_state,
                status_key=base_sample_id,
                status="failed",
                request_context=build_request_context(sample),
                error_payload=failure_payload,
            )
            client.save_status_state(status_path, status_state)
            finalize_base_sample(base_sample_id)
            continue
        for variant in variants:
            context = build_request_context(variant)
            current_status = str(status_state.get(variant["sample_id"], {}).get("status", "")).lower()
            if current_status in {"success", "skipped"}:
                skipped_count += 1
                sample_stats[base_sample_id]["skipped"] += 1
                LOGGER.info(
                    "variant_skipped sample_id=%s source_sample_id=%s status=%s",
                    variant["sample_id"],
                    context["source_sample_id"],
                    current_status,
                )
                continue
            planned_tasks.append({"base_sample_id": base_sample_id, "variant": variant})
            sample_stats[base_sample_id]["pending"] += 1
        if sample_stats[base_sample_id]["pending"] == 0:
            finalize_base_sample(base_sample_id)

    variant_progress = tqdm(total=len(planned_tasks), desc="Generating teacher supervision")
    task_iterator = iter(planned_tasks)
    in_flight: dict[Future[dict[str, Any]], dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while True:
            while len(in_flight) < max_in_flight:
                task = next(task_iterator, None)
                if task is None:
                    break
                future = executor.submit(
                    process_variant_worker,
                    task["variant"],
                    augmented_root,
                    teacher_cfg,
                    error_log_path,
                )
                in_flight[future] = task
            if not in_flight:
                break

            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                task = in_flight.pop(future)
                base_sample_id = task["base_sample_id"]
                original_variant = task["variant"]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "ok": False,
                        "stage": "worker",
                        "event": "variant_worker_failed",
                        "variant": original_variant,
                        "context": build_request_context(original_variant),
                        "error_payload": build_exception_failure_payload(exc),
                    }

                context = result["context"]
                variant = result["variant"]
                if result["ok"]:
                    record = build_output_record(variant, result["teacher_output"])
                    if record["sample_id"] not in existing_records_by_id:
                        append_jsonl(records_path, record)
                    existing_records_by_id[record["sample_id"]] = record
                    client.update_status_entry(
                        status_state,
                        status_key=variant["sample_id"],
                        status="success",
                        request_context=context,
                        output_record_path=str(records_path),
                    )
                    client.save_status_state(status_path, status_state)
                    success_count += 1
                    sample_stats[base_sample_id]["successes"] += 1
                    LOGGER.info(
                        "variant_success sample_id=%s source_sample_id=%s status=success",
                        variant["sample_id"],
                        context["source_sample_id"],
                    )
                else:
                    failure_payload = result["error_payload"]
                    event = result.get("event", "variant_worker_failed")
                    stage = result["stage"]
                    failure_count += 1
                    sample_stats[base_sample_id]["failed"] += 1
                    client.log_error(
                        event=event,
                        error=failure_payload["error"],
                        request_context=context,
                    )
                    client.update_status_entry(
                        status_state,
                        status_key=variant["sample_id"],
                        status="failed",
                        request_context=context,
                        error_payload=failure_payload,
                    )
                    client.save_status_state(status_path, status_state)
                    if stage in {"materialization", "preflight"}:
                        LOGGER.error(
                            "variant_failed sample_id=%s source_sample_id=%s stage=%s degradation_type=%s severity=%s expected_image_path=%s error=%s",
                            context.get("sample_id"),
                            context.get("source_sample_id"),
                            stage,
                            context.get("degradation_type"),
                            context.get("severity"),
                            context.get("image_path"),
                            failure_payload["error"],
                        )
                    else:
                        LOGGER.error(
                            "variant_failed sample_id=%s source_sample_id=%s stage=%s error=%s",
                            context.get("sample_id"),
                            context.get("source_sample_id"),
                            stage,
                            failure_payload["error"],
                        )

                sample_stats[base_sample_id]["pending"] -= 1
                if sample_stats[base_sample_id]["pending"] == 0:
                    finalize_base_sample(base_sample_id)
                variant_progress.update(1)

    variant_progress.close()

    records_for_analysis = list(existing_records_by_id.values())
    print_sample_analysis(records_for_analysis, args.show_first_n)
    summary_prefix = "Smoke test summary" if args.limit is not None and args.limit <= 3 else "Teacher data generation summary"
    print(
        f"{summary_prefix}: samples={len(samples)} variants_succeeded={success_count} "
        f"num_workers={num_workers} "
        f"failures={failure_count} skipped={skipped_count} output={records_path} status={status_path}"
    )
    client.close()


if __name__ == "__main__":
    main()
