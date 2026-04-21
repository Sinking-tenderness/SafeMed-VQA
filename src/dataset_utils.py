from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = ROOT / "data" / "raw" / "vqa-rad"


@dataclass
class DatasetPaths:
    dataset_root: Path
    annotations_file: Path
    images_dir: Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_vqa_rad_paths(
    dataset_root: str | Path | None = None,
    annotations_file: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> DatasetPaths:
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    candidates = [
        root / "VQA_RAD Dataset Public.json",
        root / "VQA_RAD Dataset Public.jsonl",
        root / "dataset.json",
        root / "dataset.jsonl",
    ]
    annotations = Path(annotations_file) if annotations_file else next(
        (candidate for candidate in candidates if candidate.exists()),
        None,
    )
    if annotations is None:
        raise FileNotFoundError(
            "Could not find a VQA-RAD annotation file. "
            "Place the dataset under data/raw/vqa-rad or pass --annotations-file."
        )

    image_dir_candidates = [
        root / "images",
        root / "VQA_RAD Image Folder",
        root / "VQA_RAD Image Folder" / "images",
        root,
    ]
    image_dir = Path(images_dir) if images_dir else next(
        (candidate for candidate in image_dir_candidates if candidate.exists()),
        None,
    )
    if image_dir is None:
        raise FileNotFoundError(
            "Could not find the VQA-RAD image directory. "
            "Place images under data/raw/vqa-rad/images or pass --images-dir."
        )
    return DatasetPaths(dataset_root=root, annotations_file=annotations, images_dir=image_dir)


def _normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    split = str(value).strip().lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"val", "valid", "validation", "dev"}:
        return "val"
    if split in {"test", "testing"}:
        return "test"
    return None


def _resolve_existing_image(image_root: Path, raw_value: str) -> Path:
    value = raw_value.strip()
    path = Path(value)
    direct_candidates = []
    if path.is_absolute():
        direct_candidates.append(path)
    else:
        direct_candidates.extend(
            [
                image_root / value,
                image_root / path.name,
                image_root / "images" / value,
                image_root / "images" / path.name,
            ]
        )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()
    suffix = path.suffix.lower()
    stem = path.stem
    glob_candidates = []
    if suffix:
        glob_candidates.extend(image_root.rglob(path.name))
    else:
        glob_candidates.extend(image_root.rglob(f"{stem}.*"))
    for candidate in glob_candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image path for: {raw_value}")


def _pick_first(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _make_sample_id(question: str, answer: str, image_path: Path, index: int) -> str:
    base = f"{index}|{image_path.name}|{question}|{answer}"
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"vqarad_{digest}"


def load_vqa_rad_records(
    dataset_root: str | Path | None = None,
    annotations_file: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    paths = resolve_vqa_rad_paths(dataset_root, annotations_file, images_dir)
    raw_payload = read_json(paths.annotations_file)
    if isinstance(raw_payload, dict):
        if "data" in raw_payload and isinstance(raw_payload["data"], list):
            raw_records = raw_payload["data"]
        elif "records" in raw_payload and isinstance(raw_payload["records"], list):
            raw_records = raw_payload["records"]
        else:
            raise ValueError("Unsupported VQA-RAD JSON structure.")
    elif isinstance(raw_payload, list):
        raw_records = raw_payload
    else:
        raise ValueError("Unsupported VQA-RAD annotation payload.")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        question = str(_pick_first(record, ["question", "query", "sentence"], "")).strip()
        answer = str(_pick_first(record, ["answer", "label", "response"], "")).strip()
        raw_image = _pick_first(
            record,
            ["image_name", "image_path", "image", "image_id", "image_filename"],
        )
        if not question or raw_image is None:
            continue
        image_path = _resolve_existing_image(paths.images_dir, str(raw_image))
        split = _normalize_split(_pick_first(record, ["split", "partition", "subset"]))
        sample_id = str(
            _pick_first(record, ["sample_id", "qid", "question_id"], _make_sample_id(question, answer, image_path, index))
        )
        normalized.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "question": question,
                "answer": answer,
                "split": split,
                "answer_type": _pick_first(record, ["answer_type", "question_type"]),
                "phrase_type": _pick_first(record, ["phrase_type"]),
                "raw_metadata": record,
            }
        )
    if not normalized:
        raise RuntimeError("No valid VQA-RAD samples were loaded from the dataset.")
    return normalized


def assign_splits(
    records: list[dict[str, Any]],
    val_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    explicit = [record for record in records if _normalize_split(record.get("split"))]
    if explicit and len(explicit) == len(records):
        train = [record for record in records if _normalize_split(record.get("split")) == "train"]
        val = [record for record in records if _normalize_split(record.get("split")) == "val"]
        test = [record for record in records if _normalize_split(record.get("split")) == "test"]
        return {"train": train, "val": val, "test": test}

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(round(len(shuffled) * 0.2)))
    test = shuffled[:test_size]
    remaining = shuffled[test_size:]
    val_size = int(round(len(remaining) * max(val_ratio, 0.0)))
    val = remaining[:val_size]
    train = remaining[val_size:]
    for record in train:
        record["split"] = "train"
    for record in val:
        record["split"] = "val"
    for record in test:
        record["split"] = "test"
    return {"train": train, "val": val, "test": test}


def build_index_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for record in records:
        indexed.append(
            {
                "sample_id": record["sample_id"],
                "image_path": str(Path(record["image_path"]).resolve()),
                "question": record["question"],
                "answer": record.get("answer", ""),
                "split": record["split"],
                "answer_type": record.get("answer_type"),
                "phrase_type": record.get("phrase_type"),
            }
        )
    return indexed
