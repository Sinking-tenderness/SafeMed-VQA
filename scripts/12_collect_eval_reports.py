from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, read_json, write_json


FIELDS = [
    "model_label",
    "eval_set",
    "json_success_rate",
    "answer_rate",
    "abstain_rate",
    "open_answer_accuracy",
    "abstain_quality",
    "open_utility_score",
    "teacher_decision_match_rate",
    "precise_answer_rate",
    "precise_abstain_rate",
    "over_answer_rate",
    "over_abstain_rate",
    "clean_decision_match_rate",
    "degraded_decision_match_rate",
    "paired_boundary_success_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect eval summary JSON files into CSV/Markdown/JSON tables.")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--output-csv", type=str, default="outputs/reports/eval_summary_table.csv")
    parser.add_argument("--output-md", type=str, default="outputs/reports/eval_summary_table.md")
    parser.add_argument("--output-json", type=str, default="outputs/reports/eval_summary_table.json")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def nested_get(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def first_present(payload: dict[str, Any], paths: list[list[str]]) -> Any:
    for path in paths:
        value = nested_get(payload, path)
        if value is not None:
            return value
    return None


def infer_eval_set(path: Path) -> str:
    parts = set(path.parts)
    if "eval_models_teacher_labeled" in parts:
        return "teacher_labeled"
    if "eval_models" in parts:
        return "clean"
    return path.parent.parent.name


def row_key(path: Path) -> tuple[str, str]:
    return path.parent.name, infer_eval_set(path)


def empty_row(model_label: str, eval_set: str) -> dict[str, Any]:
    row = {field: None for field in FIELDS}
    row["model_label"] = model_label
    row["eval_set"] = eval_set
    return row


def decision_rate(summary: dict[str, Any], decision: str) -> float | None:
    distribution = summary.get("decision_distribution")
    if not isinstance(distribution, dict):
        return None
    total = sum(int(value) for value in distribution.values() if isinstance(value, int))
    return (int(distribution.get(decision, 0)) / total) if total else None


def merge_model_summary(row: dict[str, Any], summary: dict[str, Any]) -> None:
    row["json_success_rate"] = summary.get("json_parse_success_rate", row["json_success_rate"])
    row["answer_rate"] = decision_rate(summary, "answer")
    row["abstain_rate"] = decision_rate(summary, "abstain")
    row["teacher_decision_match_rate"] = first_present(
        summary,
        [
            ["teacher_decision_match_rate"],
            ["decision_match_rate"],
            ["teacher_field_consistency", "decision", "match_rate"],
        ],
    )
    row["clean_decision_match_rate"] = first_present(
        summary,
        [
            ["clean_vs_degraded", "clean", "teacher_decision_metrics", "decision_match_rate"],
            ["clean_vs_degraded", "clean", "decision_match_rate"],
        ],
    )
    row["degraded_decision_match_rate"] = first_present(
        summary,
        [
            ["clean_vs_degraded", "degraded", "teacher_decision_metrics", "decision_match_rate"],
            ["clean_vs_degraded", "degraded", "decision_match_rate"],
        ],
    )


def merge_open_answer_summary(row: dict[str, Any], summary: dict[str, Any]) -> None:
    row["open_answer_accuracy"] = first_present(
        summary,
        [["open_answer_accuracy"], ["accuracy"], ["answer_accuracy"], ["metrics", "open_answer_accuracy"]],
    )
    row["abstain_quality"] = first_present(
        summary,
        [["abstain_quality"], ["abstain_quality_score"], ["metrics", "abstain_quality"]],
    )
    row["open_utility_score"] = first_present(
        summary,
        [["open_utility_score"], ["utility_score"], ["metrics", "open_utility_score"]],
    )


def merge_precise_summary(row: dict[str, Any], summary: dict[str, Any]) -> None:
    row["teacher_decision_match_rate"] = summary.get("teacher_decision_match_rate", summary.get("decision_match_rate"))
    row["precise_answer_rate"] = summary.get("precise_answer_rate")
    row["precise_abstain_rate"] = summary.get("precise_abstain_rate")
    row["over_answer_rate"] = summary.get("over_answer_rate")
    row["over_abstain_rate"] = summary.get("over_abstain_rate")
    row["clean_decision_match_rate"] = first_present(
        summary,
        [["clean_vs_degraded", "clean", "decision_match_rate"], ["clean_vs_degraded", "clean", "teacher_decision_match_rate"]],
    )
    row["degraded_decision_match_rate"] = first_present(
        summary,
        [
            ["clean_vs_degraded", "degraded", "decision_match_rate"],
            ["clean_vs_degraded", "degraded", "teacher_decision_match_rate"],
        ],
    )
    row["paired_boundary_success_rate"] = summary.get("paired_boundary_success_rate")


def collect_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    patterns = [
        "eval_models/*/summary.json",
        "eval_models/*/open_answer_judge_summary.json",
        "eval_models_teacher_labeled/*/summary.json",
        "eval_models_teacher_labeled/*/precise_abstention_summary.json",
    ]
    for pattern in patterns:
        for path in sorted(outputs_root.glob(pattern)):
            model_label, eval_set = row_key(path)
            key = (model_label, eval_set)
            rows.setdefault(key, empty_row(model_label, eval_set))
            payload = read_json(path)
            if path.name == "summary.json":
                merge_model_summary(rows[key], payload)
            elif path.name == "open_answer_judge_summary.json":
                merge_open_answer_summary(rows[key], payload)
            elif path.name == "precise_abstention_summary.json":
                merge_precise_summary(rows[key], payload)
    return [rows[key] for key in sorted(rows)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def format_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field)) for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(resolve_path(args.outputs_root))
    write_csv(resolve_path(args.output_csv), rows)
    write_markdown(resolve_path(args.output_md), rows)
    write_json(resolve_path(args.output_json), rows)
    print(json.dumps({"row_count": len(rows), "output_csv": args.output_csv, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
