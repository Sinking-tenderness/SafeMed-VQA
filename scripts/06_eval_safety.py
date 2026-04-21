from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.calibration import apply_calibration
from src.dataset_utils import ROOT, ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from src.metrics import compute_all_metrics, attach_correctness
from src.model_utils import generate_structured_output, load_processor, load_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SafeMed-VQA Pro++ safety metrics and figures.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    return parser.parse_args()


def resolve_model_path(cfg: dict) -> Path:
    final_path = ROOT / cfg["model"]["final_checkpoint"]
    if final_path.exists():
        return final_path
    return ROOT / cfg["model"]["fallback_checkpoint"]


def build_predictions(config: dict) -> list[dict]:
    model_path = resolve_model_path(config)
    processor = load_processor(str(model_path))
    model = load_student_model(
        model_name="Qwen/Qwen3-VL-8B",
        torch_dtype="bfloat16",
        adapter_path=model_path,
        trainable_adapter=False,
    )
    test_records = read_jsonl(ROOT / config["data"]["test_index"])
    calibration_payload = None
    calibration_path = ROOT / config["output"]["calibration_report"]
    if calibration_path.exists():
        calibration_payload = read_json(calibration_path)

    predictions: list[dict] = []
    for record in tqdm(test_records, desc="Running test inference"):
        prediction, _ = generate_structured_output(model, processor, record["image_path"], record["question"])
        prediction["calibrated_confidence"] = apply_calibration(prediction["raw_confidence"], calibration_payload)
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


def save_figures(records: list[dict], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ensure_dir(figures_dir)
    confidences = np.asarray(
        [
            record["prediction"].get("calibrated_confidence", record["prediction"]["raw_confidence"])
            for record in records
        ],
        dtype=np.float32,
    )
    correctness = np.asarray([float(record["is_correct"]) for record in records], dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidences, bins, right=True)
    xs = []
    ys = []
    for index in range(1, len(bins)):
        mask = bin_ids == index
        if not np.any(mask):
            continue
        xs.append(float(confidences[mask].mean()))
        ys.append(float(correctness[mask].mean()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(xs, ys, color="#005f73")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(figures_dir / "reliability_diagram.png", dpi=200)
    plt.close()

    answer_counts = {
        "answer": sum(1 for record in records if record["prediction"]["decision"] == "answer"),
        "abstain": sum(1 for record in records if record["prediction"]["decision"] == "abstain"),
    }
    plt.figure(figsize=(5, 4))
    plt.bar(answer_counts.keys(), answer_counts.values(), color=["#0a9396", "#ae2012"])
    plt.ylabel("Count")
    plt.title("Decision Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "decision_distribution.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    predictions_path = ROOT / config["output"]["test_predictions"]
    ensure_dir(predictions_path.parent)
    if predictions_path.exists():
        records = read_jsonl(predictions_path)
    else:
        records = build_predictions(config)
        write_jsonl(predictions_path, records)

    metrics = compute_all_metrics(records)
    write_json(ROOT / config["output"]["metrics_report"], metrics)
    save_figures(records, ROOT / config["output"]["figures_dir"])
    print(metrics)


if __name__ == "__main__":
    main()
