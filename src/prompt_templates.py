from __future__ import annotations

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
        "You must follow this schema exactly:\n"
        '{'
        '"explanation":"...",'
        '"raw_confidence":0.0,'
        '"decision":"answer|abstain",'
        '"abstain_type":"none|visual_insufficiency|region_missing|question_mismatch|high_risk_uncertainty",'
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


def build_teacher_messages(sample: dict[str, Any], image_url: str) -> list[dict[str, Any]]:
    if sample.get("degradation_type") in {"question_replacement", "image_replacement", "cross_sample_mismatch"}:
        extra = (
            "This sample may contain an image-question mismatch. "
            "If image evidence and question do not align, abstain with question_mismatch."
        )
    elif sample.get("degradation_type") in {"random_crop", "local_occlusion", "center_mask", "border_truncate"}:
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
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def make_file_image_url(image_path: str | Path) -> str:
    return Path(image_path).resolve().as_uri()
