from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.json_utils import parse_and_validate_json_output
from src.prompt_templates import SYSTEM_PROMPT


def _require_transformers():
    try:
        from transformers import AutoProcessor, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "transformers is required for model loading and inference. Install requirements.txt first."
        ) from exc
    return AutoProcessor, AutoTokenizer


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("peft is required for LoRA training and adapter loading. Install requirements.txt first.") from exc
    return LoraConfig, PeftModel, get_peft_model


def _resolve_model_class():
    _require_transformers()
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    if not name:
        return None
    lowered = name.lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    return None


def load_processor(model_name_or_path: str):
    AutoProcessor, _ = _require_transformers()
    return AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)


def load_tokenizer(model_name_or_path: str):
    _, AutoTokenizer = _require_transformers()
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def load_student_model(
    model_name: str,
    torch_dtype: str | None = "bfloat16",
    adapter_path: str | Path | None = None,
    trainable_adapter: bool = False,
    use_flash_attention_2: bool = False,
):
    model_cls = _resolve_model_class()
    _, PeftModel, _ = _require_peft()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _dtype_from_name(torch_dtype),
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = model_cls.from_pretrained(model_name, **kwargs)
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable_adapter)
    return model


def create_lora_config(config: dict[str, Any]):
    LoraConfig, _, _ = _require_peft()
    return LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(config.get("target_modules", [])),
    )


def _is_linear_module(module: Any) -> bool:
    return isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "Linear"


def _target_matches_module(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}") or module_name.endswith(target)


def _categorize_lora_module(module_name: str) -> str:
    if ".visual.blocks." in module_name or module_name.startswith("visual.blocks."):
        return "visual_blocks"
    if ".visual.deepstack_merger_list." in module_name or module_name.startswith("visual.deepstack_merger_list."):
        return "visual_deepstack_merger"
    if ".visual.merger." in module_name or module_name.startswith("visual.merger."):
        return "visual_merger"
    if ".language_model." in module_name or module_name.startswith("language_model."):
        return "language_model"
    return "other"


def resolve_lora_target_modules(model: Any, lora_config: dict[str, Any]) -> list[str]:
    target_regex = list(lora_config.get("target_module_regex") or [])
    if not target_regex:
        return list(lora_config.get("target_modules", []))

    patterns = [re.compile(pattern) for pattern in target_regex]
    matched: list[str] = []
    for module_name, module in model.named_modules():
        if not module_name or not _is_linear_module(module):
            continue
        if any(pattern.fullmatch(module_name) or pattern.match(module_name) for pattern in patterns):
            matched.append(module_name)
    return matched


def inspect_lora_target_modules(
    model: Any,
    target_modules: list[str],
    *,
    require_visual_merger: bool = False,
    max_print: int = 50,
) -> dict[str, Any]:
    matched: list[str] = []
    for module_name, module in model.named_modules():
        if not module_name or not _is_linear_module(module):
            continue
        if any(_target_matches_module(module_name, target) for target in target_modules):
            matched.append(module_name)

    category_counts = {
        "language_model": 0,
        "visual_merger": 0,
        "visual_deepstack_merger": 0,
        "visual_blocks": 0,
        "other": 0,
    }
    for module_name in matched:
        category_counts[_categorize_lora_module(module_name)] += 1

    summary = {
        "total_matched_modules": len(matched),
        "language_matched_modules": category_counts["language_model"],
        "visual_merger_matched_modules": category_counts["visual_merger"],
        "visual_deepstack_merger_matched_modules": category_counts["visual_deepstack_merger"],
        "visual_blocks_matched_modules": category_counts["visual_blocks"],
        "other_matched_modules": category_counts["other"],
        "matched_module_names": matched,
    }
    print(
        "LoRA target module summary: "
        f"total={summary['total_matched_modules']} "
        f"language={summary['language_matched_modules']} "
        f"visual_merger={summary['visual_merger_matched_modules']} "
        f"visual_deepstack_merger={summary['visual_deepstack_merger_matched_modules']} "
        f"visual_blocks={summary['visual_blocks_matched_modules']} "
        f"other={summary['other_matched_modules']}"
    )
    print(f"First {min(max_print, len(matched))} matched LoRA target modules:")
    for module_name in matched[:max_print]:
        print(module_name)

    if not matched:
        raise RuntimeError("No LoRA target modules matched the model. Refusing to start training.")
    if require_visual_merger and category_counts["visual_merger"] + category_counts["visual_deepstack_merger"] == 0:
        raise RuntimeError("No visual merger/deepstack merger LoRA targets matched. Refusing to start training.")
    return summary


def maybe_apply_lora(model, lora_config: dict[str, Any]):
    _, _, get_peft_model = _require_peft()
    if hasattr(model, "peft_config") and getattr(model, "peft_config", None):
        return model
    resolved_config = dict(lora_config)
    resolved_targets = resolve_lora_target_modules(model, resolved_config)
    resolved_config["target_modules"] = resolved_targets
    require_visual_merger = bool(resolved_config.get("require_visual_merger_targets", False))
    inspect_lora_target_modules(model, resolved_targets, require_visual_merger=require_visual_merger)
    return get_peft_model(model, create_lora_config(resolved_config))


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def build_training_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    assistant_json = json.dumps(record["teacher_output"], ensure_ascii=False)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the medical image and answer the question in strict JSON.\n"
                        f"Question: {record['question']}"
                    ),
                },
                {"type": "image"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_json}]},
    ]


@dataclass
class MedicalSFTDataCollator:
    processor: Any
    max_length: int = 1536

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images = [load_image(feature["image_path"]) for feature in features]
        messages = [feature["messages"] for feature in features]
        prompt_messages = [message[:-1] for message in messages]

        prompt_texts = [
            self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in prompt_messages
        ]
        full_texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, : int(prompt_length)] = -100
            labels[index, model_inputs["input_ids"][index] == pad_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def prepare_sft_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        prepared.append(
            {
                **record,
                "messages": build_training_messages(record),
            }
        )
    return prepared


def build_inference_messages(question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the medical image and answer the question in strict JSON.\n"
                        "First audit the image evidence, then answer or abstain consistently.\n"
                        f"Question: {question}"
                    ),
                },
                {"type": "image"},
            ],
        },
    ]


@torch.inference_mode()
def generate_structured_output(
    model,
    processor,
    image_path: str | Path,
    question: str,
    max_new_tokens: int = 512,
) -> tuple[dict[str, Any], str]:
    messages = build_inference_messages(question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = load_image(image_path)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if hasattr(model, "device") and model.device.type != "cpu":
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    completion = generated[0][prompt_len:]
    decoded = processor.tokenizer.decode(completion, skip_special_tokens=True).strip()
    return parse_and_validate_json_output(decoded), decoded


def prepare_preference_texts(
    processor,
    question: str,
    image_path: str | Path,
    completion_text: str,
    max_length: int | None = None,
) -> tuple[dict[str, Any], int]:
    messages = build_inference_messages(question)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + completion_text
    image = load_image(image_path)
    processor_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        processor_kwargs.update({"truncation": True, "max_length": max_length})
    full_inputs = processor(text=[full_text], images=[image], **processor_kwargs)
    prompt_inputs = processor(text=[prompt_text], images=[image], **processor_kwargs)
    prompt_len = int(prompt_inputs["attention_mask"].sum().item())
    return full_inputs, prompt_len


def move_batch_to_device(batch: dict[str, Any], device: torch.device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def completion_logprob(model, batch: dict[str, Any], prompt_len: int) -> torch.Tensor:
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    completion_mask = positions >= max(prompt_len - 1, 0)
    if model.config.pad_token_id is not None:
        completion_mask &= labels != model.config.pad_token_id
    masked_log_probs = token_log_probs * completion_mask
    return masked_log_probs.sum(dim=1)
