from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "raw" / "vqa-rad"
IMAGES_DIR = OUT_ROOT / "images"
ANNOTATIONS_PATH = OUT_ROOT / "VQA_RAD Dataset Public.json"

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("flaviagiammarino/vqa-rad")

    records = []
    for split_name in dataset.keys():
        split = dataset[split_name]
        for idx, example in enumerate(tqdm(split, desc=f"exporting {split_name}")):
            image = example["image"]
            question = str(example["question"]).strip()
            answer = str(example["answer"]).strip()

            image_name = f"{split_name}_{idx:06d}.png"
            image_path = IMAGES_DIR / image_name
            image.save(image_path)

            records.append(
                {
                    "question": question,
                    "answer": answer,
                    "image_name": image_name,
                    "split": split_name,
                }
            )

    with ANNOTATIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote annotations to: {ANNOTATIONS_PATH}")
    print(f"Saved images to: {IMAGES_DIR}")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    main()
