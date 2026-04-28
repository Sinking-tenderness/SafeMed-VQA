from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float) -> torch.Tensor:
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return -torch.nn.functional.logsigmoid(logits).mean()


def resolve_config_path(config: dict, *path: str, default=None):
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    train_jsonl = data_cfg.get("train_jsonl", data_cfg.get("train_pairs_jsonl"))
    if not train_jsonl:
        raise KeyError("DPO config requires data.train_jsonl or data.train_pairs_jsonl.")
    pairs = read_jsonl(ROOT / train_jsonl)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No DPO pairs found. Run scripts/03_build_dpo_pairs.py first.")

    model_name = model_cfg["model_name"]
    torch_dtype = str(model_cfg.get("torch_dtype", "bfloat16"))
    adapter_path = model_cfg.get("sft_adapter_path", model_cfg.get("sft_checkpoint"))
    use_flash_attention_2 = bool(model_cfg.get("use_flash_attention_2", False))

    processor = load_processor(model_name)
    policy_model = load_student_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        adapter_path=ROOT / adapter_path if adapter_path else None,
        trainable_adapter=True,
        use_flash_attention_2=use_flash_attention_2,
    )
    reference_model = load_student_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        adapter_path=ROOT / adapter_path if adapter_path else None,
        trainable_adapter=False,
        use_flash_attention_2=use_flash_attention_2,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    device = next(policy_model.parameters()).device
    learning_rate = float(train_cfg.get("learning_rate", 5e-6))
    optimizer = AdamW(policy_model.parameters(), lr=learning_rate)
    beta = float(resolve_config_path(config, "dpo", "beta", default=train_cfg.get("beta", 0.1)))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    epochs = int(train_cfg.get("num_train_epochs", 1))
    max_seq_length = int(train_cfg.get("max_seq_length", 3072))
    output_dir_value = train_cfg.get("output_dir", model_cfg.get("output_dir"))
    if not output_dir_value:
        raise KeyError("DPO config requires training.output_dir or model.output_dir.")
    output_dir = ROOT / output_dir_value
    print(
        "DPO training config summary: "
        f"pairs={len(pairs)} "
        f"epochs={epochs} "
        f"grad_accum={grad_accum} "
        f"beta={beta} "
        f"learning_rate={learning_rate} "
        f"max_seq_length={max_seq_length} "
        f"output_dir={output_dir}"
    )

    policy_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, pair in enumerate(tqdm(pairs, desc=f"DPO epoch {epoch + 1}/{epochs}"), start=1):
            chosen_inputs, chosen_prompt_len = prepare_preference_texts(
                processor, pair["question"], pair["image_path"], pair["chosen"], max_length=max_seq_length
            )
            rejected_inputs, rejected_prompt_len = prepare_preference_texts(
                processor, pair["question"], pair["image_path"], pair["rejected"], max_length=max_seq_length
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

    ensure_dir(output_dir)
    policy_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
