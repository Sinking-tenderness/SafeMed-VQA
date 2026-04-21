from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from src.dataset_utils import ROOT, ensure_dir, read_jsonl
from src.model_utils import (
    completion_logprob,
    load_processor,
    load_student_model,
    move_batch_to_device,
    prepare_preference_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-scale DPO training on high-risk pairs.")
    parser.add_argument("--config", type=str, default="configs/dpo.yaml")
    return parser.parse_args()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float) -> torch.Tensor:
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return -torch.nn.functional.logsigmoid(logits).mean()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    train_cfg = config["training"]
    pairs = read_jsonl(ROOT / config["data"]["train_pairs_jsonl"])
    if not pairs:
        raise RuntimeError("No DPO pairs found. Run scripts/03_build_dpo_pairs.py first.")

    processor = load_processor(model_cfg["model_name"])
    policy_model = load_student_model(
        model_name=model_cfg["model_name"],
        torch_dtype="bfloat16",
        adapter_path=ROOT / model_cfg["sft_checkpoint"],
        trainable_adapter=True,
    )
    reference_model = load_student_model(
        model_name=model_cfg["model_name"],
        torch_dtype="bfloat16",
        adapter_path=ROOT / model_cfg["sft_checkpoint"],
        trainable_adapter=False,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    device = next(policy_model.parameters()).device
    optimizer = AdamW(policy_model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-6)))
    beta = float(train_cfg.get("beta", 0.1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    epochs = int(train_cfg.get("num_train_epochs", 1))

    policy_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, pair in enumerate(tqdm(pairs, desc=f"DPO epoch {epoch + 1}/{epochs}"), start=1):
            chosen_inputs, chosen_prompt_len = prepare_preference_texts(
                processor, pair["question"], pair["image_path"], pair["chosen"]
            )
            rejected_inputs, rejected_prompt_len = prepare_preference_texts(
                processor, pair["question"], pair["image_path"], pair["rejected"]
            )

            chosen_inputs = move_batch_to_device(chosen_inputs, device)
            rejected_inputs = move_batch_to_device(rejected_inputs, device)

            policy_chosen = completion_logprob(policy_model, chosen_inputs, chosen_prompt_len)
            policy_rejected = completion_logprob(policy_model, rejected_inputs, rejected_prompt_len)
            with torch.no_grad():
                ref_chosen = completion_logprob(reference_model, chosen_inputs, chosen_prompt_len)
                ref_rejected = completion_logprob(reference_model, rejected_inputs, rejected_prompt_len)

            loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta) / grad_accum
            loss.backward()
            running_loss += float(loss.item())

            if step % grad_accum == 0 or step == len(pairs):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch + 1}: loss={running_loss:.4f}")

    output_dir = ROOT / model_cfg["output_dir"]
    ensure_dir(output_dir)
    policy_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
