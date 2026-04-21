from __future__ import annotations

from typing import Any

import numpy as np


def collect_confidence_targets(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    confidences: list[float] = []
    targets: list[float] = []
    for record in records:
        prediction = record.get("prediction", {})
        raw_confidence = prediction.get("raw_confidence")
        is_correct = record.get("is_correct")
        if raw_confidence is None or is_correct is None:
            continue
        confidences.append(float(raw_confidence))
        targets.append(float(bool(is_correct)))
    if not confidences:
        raise ValueError("No confidence/target pairs available for calibration.")
    return np.asarray(confidences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def fit_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("scikit-learn is required for confidence calibration. Install requirements.txt first.") from exc

    confidences, targets = collect_confidence_targets(records)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(confidences, targets)
    calibrated = model.predict(confidences)
    return {
        "method": "isotonic_regression",
        "x_thresholds": [float(value) for value in model.X_thresholds_],
        "y_thresholds": [float(value) for value in model.y_thresholds_],
        "num_samples": int(len(confidences)),
        "mean_raw_confidence": float(confidences.mean()),
        "mean_calibrated_confidence": float(np.mean(calibrated)),
    }


def apply_calibration(raw_confidence: float, calibration_payload: dict[str, Any] | None) -> float:
    if not calibration_payload or calibration_payload.get("method") != "isotonic_regression":
        return float(raw_confidence)
    x = np.asarray(calibration_payload["x_thresholds"], dtype=np.float32)
    y = np.asarray(calibration_payload["y_thresholds"], dtype=np.float32)
    clipped = np.clip(float(raw_confidence), float(x.min()), float(x.max()))
    return float(np.interp(clipped, x, y))
