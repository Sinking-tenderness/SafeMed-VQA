from __future__ import annotations

import json
from typing import Any


REQUIRED_FIELDS = {
    "explanation",
    "raw_confidence",
    "decision",
    "abstain_type",
    "risk_level",
    "answer",
}
ALLOWED_DECISIONS = {"answer", "abstain"}
ALLOWED_ABSTAIN_TYPES = {
    "none",
    "visual_insufficiency",
    "region_missing",
    "question_mismatch",
    "high_risk_uncertainty",
}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


def extract_json_string(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json", "", 1).strip()
    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for index in range(start, len(candidate)):
        char = candidate[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    raise ValueError("Unterminated JSON object in model output.")


def parse_json_output(text: str) -> dict[str, Any]:
    return json.loads(extract_json_string(text))


def _is_float_like(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def validate_output_schema(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(payload)
    if missing:
        errors.append(f"missing_fields={sorted(missing)}")
        return errors

    if not isinstance(payload["explanation"], str) or not payload["explanation"].strip():
        errors.append("explanation must be a non-empty string")
    if not _is_float_like(payload["raw_confidence"]):
        errors.append("raw_confidence must be a number")
    else:
        confidence = float(payload["raw_confidence"])
        if confidence < 0.0 or confidence > 1.0:
            errors.append("raw_confidence must be within [0, 1]")
    if payload["decision"] not in ALLOWED_DECISIONS:
        errors.append(f"decision must be one of {sorted(ALLOWED_DECISIONS)}")
    if payload["abstain_type"] not in ALLOWED_ABSTAIN_TYPES:
        errors.append(f"abstain_type must be one of {sorted(ALLOWED_ABSTAIN_TYPES)}")
    if payload["risk_level"] not in ALLOWED_RISK_LEVELS:
        errors.append(f"risk_level must be one of {sorted(ALLOWED_RISK_LEVELS)}")
    if not isinstance(payload["answer"], str) or not payload["answer"].strip():
        errors.append("answer must be a non-empty string")

    explanation = payload.get("explanation", "").lower()
    decision = payload.get("decision")
    abstain_type = payload.get("abstain_type")
    if decision == "answer" and abstain_type != "none":
        errors.append("abstain_type must be none when decision=answer")
    if decision == "abstain" and abstain_type == "none":
        errors.append("abstain_type cannot be none when decision=abstain")
    if decision == "answer" and any(token in explanation for token in ["not clear", "insufficient", "cannot assess", "look unclear"]):
        errors.append("explanation indicates insufficient evidence while decision=answer")
    if decision == "abstain" and payload.get("answer", "").strip().lower() in {"normal", "yes", "no"}:
        errors.append("answer is overly definite while decision=abstain")
    return errors


def normalize_output(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["explanation"] = str(normalized["explanation"]).strip()
    normalized["raw_confidence"] = float(normalized["raw_confidence"])
    normalized["decision"] = str(normalized["decision"]).strip().lower()
    normalized["abstain_type"] = str(normalized["abstain_type"]).strip().lower()
    normalized["risk_level"] = str(normalized["risk_level"]).strip().lower()
    normalized["answer"] = str(normalized["answer"]).strip()
    errors = validate_output_schema(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def parse_and_validate_json_output(text: str) -> dict[str, Any]:
    return normalize_output(parse_json_output(text))
