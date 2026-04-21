from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return str(target.resolve())


def apply_gaussian_blur(image: Image.Image, severity: str) -> Image.Image:
    radius_map = {"low": 1.2, "medium": 2.4, "high": 4.0}
    return image.filter(ImageFilter.GaussianBlur(radius=radius_map.get(severity, 2.4)))


def apply_speckle_noise(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    sigma_map = {"low": 0.05, "medium": 0.12, "high": 0.2}
    sigma = sigma_map.get(severity, 0.12)
    array = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0.0, sigma, array.shape)
    speckled = np.clip(array + array * noise, 0.0, 1.0)
    return Image.fromarray((speckled * 255).astype(np.uint8))


def apply_resolution_drop(image: Image.Image, severity: str) -> Image.Image:
    scale_map = {"low": 0.8, "medium": 0.55, "high": 0.35}
    scale = scale_map.get(severity, 0.55)
    width, height = image.size
    downsampled = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    return downsampled.resize((width, height), Image.Resampling.BICUBIC)


def apply_contrast_brightness_shift(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    span_map = {"low": 0.12, "medium": 0.22, "high": 0.34}
    span = span_map.get(severity, 0.22)
    contrast_factor = 1.0 + rng.uniform(-span, span)
    brightness_factor = 1.0 + rng.uniform(-span, span)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return image


def apply_random_crop(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    crop_ratio_map = {"low": 0.92, "medium": 0.8, "high": 0.65}
    ratio = crop_ratio_map.get(severity, 0.8)
    width, height = image.size
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def apply_occlusion(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    coverage_map = {"low": 0.12, "medium": 0.22, "high": 0.35}
    coverage = coverage_map.get(severity, 0.22)
    width, height = image.size
    occ_w = max(1, int(width * coverage))
    occ_h = max(1, int(height * coverage))
    left = rng.randint(0, max(0, width - occ_w))
    top = rng.randint(0, max(0, height - occ_h))
    occluded = image.copy()
    patch = Image.new("RGB", (occ_w, occ_h), color=(0, 0, 0))
    occluded.paste(patch, (left, top))
    return occluded


def apply_center_mask(image: Image.Image, severity: str) -> Image.Image:
    ratio_map = {"low": 0.18, "medium": 0.3, "high": 0.42}
    ratio = ratio_map.get(severity, 0.3)
    width, height = image.size
    mask_w = int(width * ratio)
    mask_h = int(height * ratio)
    left = (width - mask_w) // 2
    top = (height - mask_h) // 2
    masked = image.copy()
    masked.paste(Image.new("RGB", (mask_w, mask_h), color=(0, 0, 0)), (left, top))
    return masked


def apply_border_truncate(image: Image.Image, severity: str) -> Image.Image:
    ratio_map = {"low": 0.08, "medium": 0.15, "high": 0.22}
    ratio = ratio_map.get(severity, 0.15)
    width, height = image.size
    left = int(width * ratio)
    top = int(height * ratio)
    cropped = image.crop((left, top, width, height))
    return ImageOps.pad(cropped, (width, height), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def severity_choices(weights: dict[str, float] | None = None) -> list[str]:
    if not weights:
        return ["low", "medium", "high"]
    expanded: list[str] = []
    for label, value in weights.items():
        expanded.extend([label] * max(1, int(value * 100)))
    return expanded or ["low", "medium", "high"]


def apply_degradation(
    image: Image.Image,
    degradation_type: str,
    severity: str,
    rng: random.Random,
) -> Image.Image:
    if degradation_type == "gaussian_blur":
        return apply_gaussian_blur(image, severity)
    if degradation_type == "speckle_noise":
        return apply_speckle_noise(image, severity, rng)
    if degradation_type == "resolution_drop":
        return apply_resolution_drop(image, severity)
    if degradation_type == "contrast_brightness_shift":
        return apply_contrast_brightness_shift(image, severity, rng)
    if degradation_type == "random_crop":
        return apply_random_crop(image, severity, rng)
    if degradation_type == "local_occlusion":
        return apply_occlusion(image, severity, rng)
    if degradation_type == "center_mask":
        return apply_center_mask(image, severity)
    if degradation_type == "border_truncate":
        return apply_border_truncate(image, severity)
    raise ValueError(f"Unsupported degradation_type: {degradation_type}")


def construct_mismatch_sample(
    sample: dict[str, Any],
    pool: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    mismatch_type = rng.choice(["question_replacement", "image_replacement", "cross_sample_mismatch"])
    other = sample
    while other["sample_id"] == sample["sample_id"]:
        other = rng.choice(pool)
    mismatched = dict(sample)
    if mismatch_type == "question_replacement":
        mismatched["question"] = other["question"]
    elif mismatch_type == "image_replacement":
        mismatched["image_path"] = other["image_path"]
    else:
        mismatched["question"] = other["question"]
        third = other
        while third["sample_id"] in {sample["sample_id"], other["sample_id"]}:
            third = rng.choice(pool)
        mismatched["image_path"] = third["image_path"]
    mismatched["is_counterfactual"] = True
    mismatched["degradation_type"] = mismatch_type
    mismatched["source_sample_id"] = sample["sample_id"]
    mismatched["severity"] = "high"
    return mismatched


def create_augmented_sample(
    sample: dict[str, Any],
    pool: list[dict[str, Any]],
    output_dir: str | Path,
    severity: str,
    degradation_type: str,
    rng: random.Random,
) -> dict[str, Any]:
    if degradation_type in {"question_replacement", "image_replacement", "cross_sample_mismatch"}:
        return construct_mismatch_sample(sample, pool, rng)

    image = load_image(sample["image_path"])
    augmented = apply_degradation(image, degradation_type, severity, rng)
    suffix = Path(sample["image_path"]).suffix or ".png"
    out_path = Path(output_dir) / f"{sample['sample_id']}_{degradation_type}_{severity}{suffix}"
    saved_path = save_image(augmented, out_path)
    return {
        **sample,
        "sample_id": f"{sample['sample_id']}__{degradation_type}_{severity}",
        "image_path": saved_path,
        "is_counterfactual": True,
        "degradation_type": degradation_type,
        "severity": severity,
        "source_sample_id": sample["sample_id"],
    }
