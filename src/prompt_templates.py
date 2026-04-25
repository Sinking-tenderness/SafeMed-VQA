from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a cautious medical vision-language teacher. "
    "Always output valid JSON only. "
    "First audit image evidence quality and visibility, then reason about the question. "
    "When evidence is insufficient, prefer abstain."
)


def _base_user_instruction(sample: dict[str, Any]) -> str:
    return (
        "Analyze the medical image and answer the question in strict JSON.\n"
        "Return JSON only. Do not use markdown code fences or extra prose.\n"
        'If decision="abstain", answer must still be a short safe refusal statement and must not be empty.\n'
        "You must follow this schema exactly:\n"
        '{'
        '"explanation":"...",'
        '"raw_confidence":0.0,'
        '"decision":"answer|abstain",'
        '"abstain_type":"none|visual_insufficiency|region_missing|high_risk_uncertainty",'
        '"risk_level":"low|medium|high",'
        '"answer":"..."'
        '}\n'
        f"Question: {sample['question']}"
    )


def _sample_context(sample: dict[str, Any]) -> str:
    if not sample.get("is_counterfactual"):
        return "Sample type: original training sample."
    degradation_type = sample.get("degradation_type", "unknown")
    severity = sample.get("severity", "unknown")
    return f"Sample type: counterfactual sample. degradation_type={degradation_type}; severity={severity}."


def build_teacher_messages(sample: dict[str, Any], image_url: str | dict[str, Any]) -> list[dict[str, Any]]:
    if sample.get("degradation_type") in {"random_crop", "local_occlusion", "center_mask", "border_truncate"}:
        extra = (
            "This sample may hide clinically relevant regions. "
            "If key anatomy is missing, abstain with region_missing."
        )
    elif sample.get("is_counterfactual"):
        extra = (
            "This sample may be visually degraded. "
            "If the evidence remains sufficient, answering is allowed; otherwise abstain with visual_insufficiency "
            "or high_risk_uncertainty."
        )
    else:
        extra = "For standard samples, answer when evidence is sufficient and remain concise."

    user_text = "\n".join([_base_user_instruction(sample), _sample_context(sample), extra])
    image_part = image_url
    if isinstance(image_url, str):
        image_part = {"type": "image_url", "image_url": {"url": image_url}}
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                image_part,
            ],
        },
    ]


def build_analysis_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "degradation_type": record.get("degradation_type", "none"),
        "question": record.get("question"),
        "teacher_output": record.get("teacher_output"),
    }


def make_file_image_url(image_path: str | Path, transport: str = "data_url") -> str:
    path_text = str(image_path)
    if path_text.startswith(("http://", "https://", "data:")):
        return path_text
    if transport != "data_url":
        raise ValueError(f"Unsupported image transport: {transport}")

    source = Path(image_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    mime_type, _ = mimetypes.guess_type(source.name)
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"
