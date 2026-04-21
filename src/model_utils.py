from __future__ import annotations

import json
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


def maybe_apply_lora(model, lora_config: dict[str, Any]):
    _, _, get_peft_model = _require_peft()
    if hasattr(model, "peft_config") and getattr(model, "peft_config", None):
        return model
    return get_peft_model(model, create_lora_config(lora_config))


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
) -> tuple[dict[str, Any], int]:
    messages = build_inference_messages(question)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + completion_text
    image = load_image(image_path)
    full_inputs = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
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
