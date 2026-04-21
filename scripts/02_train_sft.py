from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.dataset_utils import ROOT, read_jsonl
from src.model_utils import (
    MedicalSFTDataCollator,
    load_processor,
    load_student_model,
    maybe_apply_lora,
    prepare_sft_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SafeMed-VQA Pro++ student with LoRA SFT.")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    records = prepare_sft_records(read_jsonl(ROOT / data_cfg["train_jsonl"]))
    dataset = Dataset.from_list(records)
    eval_ratio = float(data_cfg.get("eval_ratio", 0.05))
    if eval_ratio > 0.0 and len(dataset) > 20:
        split_dataset = dataset.train_test_split(test_size=eval_ratio, seed=int(config.get("seed", 42)))
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    processor = load_processor(model_cfg["model_name"])
    model = load_student_model(
        model_cfg["model_name"],
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        use_flash_attention_2=bool(model_cfg.get("use_flash_attention_2", False)),
    )
    model = maybe_apply_lora(model, config["lora"])
    collator = MedicalSFTDataCollator(
        processor=processor,
        max_length=int(train_cfg.get("max_seq_length", 1536)),
    )

    training_args = SFTConfig(
        output_dir=str(ROOT / train_cfg["output_dir"]),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 100)),
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=list(train_cfg.get("report_to", ["wandb"])),
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        eval_strategy="steps" if eval_dataset is not None else "no",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(ROOT / train_cfg["output_dir"])


if __name__ == "__main__":
    main()
