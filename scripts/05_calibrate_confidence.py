from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.calibration import apply_calibration, fit_calibration
from src.dataset_utils import ROOT, ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from src.metrics import attach_correctness
from src.model_utils import generate_structured_output, load_processor, load_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit confidence calibration on validation predictions.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    return parser.parse_args()


def resolve_model_path(cfg: dict) -> Path:
    final_path = ROOT / cfg["model"]["final_checkpoint"]
    if final_path.exists():
        return final_path
    return ROOT / cfg["model"]["fallback_checkpoint"]


def generate_validation_predictions(config: dict) -> list[dict]:
    model_path = resolve_model_path(config)
    processor = load_processor(str(model_path))
    model = load_student_model(
        model_name="Qwen/Qwen3-VL-8B",
        torch_dtype="bfloat16",
        adapter_path=model_path,
        trainable_adapter=False,
    )
    val_records = read_jsonl(ROOT / config["data"]["val_index"])
    predictions: list[dict] = []
    for record in tqdm(val_records, desc="Running validation inference"):
        prediction, _ = generate_structured_output(model, processor, record["image_path"], record["question"])
        is_correct = attach_correctness(prediction, record.get("answer", ""), ground_truth_should_abstain=False)
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "question": record["question"],
                "image_path": record["image_path"],
                "reference_answer": record.get("answer", ""),
                "ground_truth_should_abstain": False,
                "prediction": prediction,
                "is_correct": is_correct,
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    val_predictions_path = ROOT / config["output"]["val_predictions"]
    ensure_dir(val_predictions_path.parent)
    if val_predictions_path.exists():
        predictions = read_jsonl(val_predictions_path)
    else:
        predictions = generate_validation_predictions(config)
        write_jsonl(val_predictions_path, predictions)

    calibration_payload = fit_calibration(predictions)
    for record in predictions:
        record["prediction"]["calibrated_confidence"] = apply_calibration(
            record["prediction"]["raw_confidence"], calibration_payload
        )
    write_jsonl(val_predictions_path, predictions)
    write_json(ROOT / config["output"]["calibration_report"], calibration_payload)
    print(json.dumps(calibration_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
