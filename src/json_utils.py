from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)
ABSTAIN_ANSWER_FALLBACK = "I cannot answer safely based on the available image evidence."
MAX_FAILURE_PREVIEW_CHARS = 500

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


def _repair_abstain_answer(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    repaired = dict(payload)
    decision = str(repaired.get("decision", "")).strip().lower()
    answer = repaired.get("answer")
    if decision == "abstain" and (answer is None or not str(answer).strip()):
        repaired["answer"] = ABSTAIN_ANSWER_FALLBACK
        LOGGER.warning("Applied abstain-answer fallback repair to teacher output.")
        return repaired, True
    return repaired, False


def normalize_output(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Teacher output must be a JSON object.")
    repaired_payload, _ = _repair_abstain_answer(payload)
    missing = REQUIRED_FIELDS - set(repaired_payload)
    if missing:
        raise ValueError(f"missing_fields={sorted(missing)}")

    normalized = dict(repaired_payload)
    normalized["explanation"] = str(normalized["explanation"]).strip()
    try:
        normalized["raw_confidence"] = float(normalized["raw_confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_confidence must be parseable as a number") from exc
    normalized["decision"] = str(normalized["decision"]).strip().lower()
    normalized["abstain_type"] = str(normalized["abstain_type"]).strip().lower()
    normalized["risk_level"] = str(normalized["risk_level"]).strip().lower()
    normalized["answer"] = str(normalized["answer"]).strip()
    errors = validate_output_schema(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def build_output_failure_payload(text: str | None, error: Exception) -> dict[str, Any]:
    preview = None
    if text is not None:
        preview = str(text).strip()[:MAX_FAILURE_PREVIEW_CHARS] or None
    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "response_preview": preview,
    }


def build_exception_failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "response_preview": None,
    }


def safe_parse_and_validate_json_output(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return parse_and_validate_json_output(text), None
    except Exception as exc:  # noqa: BLE001
        return None, build_output_failure_payload(text, exc)


def parse_and_validate_json_output(text: str) -> dict[str, Any]:
    return normalize_output(parse_json_output(text))
