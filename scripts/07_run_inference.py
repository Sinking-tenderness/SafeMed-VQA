from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.calibration import apply_calibration
from src.dataset_utils import ROOT, read_json, read_jsonl, write_jsonl
from src.model_utils import generate_structured_output, load_processor, load_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final SafeMed-VQA Pro++ inference.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--input-jsonl", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default="outputs/predictions/inference_predictions.jsonl")
    parser.add_argument("--apply-calibration", action="store_true")
    return parser.parse_args()


def resolve_model_path(cfg: dict) -> Path:
    final_path = ROOT / cfg["model"]["final_checkpoint"]
    if final_path.exists():
        return final_path
    return ROOT / cfg["model"]["fallback_checkpoint"]


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not args.input_jsonl and not (args.image_path and args.question):
        raise ValueError("Provide either --input-jsonl or both --image-path and --question.")

    model_path = resolve_model_path(config)
    processor = load_processor(str(model_path))
    model = load_student_model(
        model_name="Qwen/Qwen3-VL-8B",
        torch_dtype="bfloat16",
        adapter_path=model_path,
        trainable_adapter=False,
    )

    calibration_payload = None
    if args.apply_calibration:
        calibration_path = ROOT / config["output"]["calibration_report"]
        if calibration_path.exists():
            calibration_payload = read_json(calibration_path)

    if args.input_jsonl:
        items = read_jsonl(ROOT / args.input_jsonl)
        outputs = []
        for item in items:
            prediction, _ = generate_structured_output(model, processor, item["image_path"], item["question"])
            if calibration_payload:
                prediction["calibrated_confidence"] = apply_calibration(
                    prediction["raw_confidence"], calibration_payload
                )
            outputs.append(
                {
                    "sample_id": item.get("sample_id"),
                    "image_path": item["image_path"],
                    "question": item["question"],
                    "prediction": prediction,
                }
            )
        write_jsonl(ROOT / args.output_jsonl, outputs)
        print(f"Wrote {len(outputs)} predictions to {ROOT / args.output_jsonl}")
        return

    prediction, _ = generate_structured_output(model, processor, args.image_path, args.question)
    if calibration_payload:
        prediction["calibrated_confidence"] = apply_calibration(prediction["raw_confidence"], calibration_payload)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
