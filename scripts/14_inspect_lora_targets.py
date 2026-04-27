from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_utils import inspect_lora_target_modules, resolve_lora_target_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run inspect LoRA target modules for SFT configs.")
    parser.add_argument("--config", type=str, default="configs/sft_lora_merger.yaml")
    parser.add_argument("--base-model-path", type=str, default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    return parser.parse_args()


def load_cpu_model(base_model_path: str):
    try:
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415

        model_cls = AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq  # noqa: PLC0415

            model_cls = AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM  # noqa: PLC0415

            model_cls = AutoModelForCausalLM
    return model_cls.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype="auto")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = load_cpu_model(args.base_model_path)
    lora_cfg = config["lora"]
    resolved_targets = resolve_lora_target_modules(model, lora_cfg)
    summary = inspect_lora_target_modules(
        model,
        resolved_targets,
        require_visual_merger=True,
        max_print=100,
    )
    if summary["visual_merger_matched_modules"] + summary["visual_deepstack_merger_matched_modules"] == 0:
        raise RuntimeError("No visual merger/deepstack merger targets matched; 不允许开始训练。")


if __name__ == "__main__":
    main()
