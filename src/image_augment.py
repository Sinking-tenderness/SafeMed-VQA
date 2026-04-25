from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SUPPORTED_DEGRADATION_TYPES = [
    "gaussian_blur",
    "speckle_noise",
    "resolution_drop",
    "contrast_brightness_shift",
    "random_crop",
    "local_occlusion",
    "center_mask",
    "border_truncate",
]

DEGRADATION_TYPE_ALIASES = {
    "gaussian_noise": "speckle_noise",
    "brightness_shift": "contrast_brightness_shift",
    "contrast_shift": "contrast_brightness_shift",
    "low_resolution": "resolution_drop",
    "occlusion": "local_occlusion",
    "crop": "random_crop",
    "border_truncation": "border_truncate",
}

SEVERITY_ALIASES = {
    "mild": "low",
    "severe": "high",
}


def normalize_severity(severity: str) -> str:
    normalized = str(severity).strip().lower()
    return SEVERITY_ALIASES.get(normalized, normalized)


def normalize_degradation_type(degradation_type: str) -> str:
    normalized = str(degradation_type).strip().lower()
    return DEGRADATION_TYPE_ALIASES.get(normalized, normalized)


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
    degraded, _ = apply_degradation_with_params(image, degradation_type, severity, rng)
    return degraded


def apply_degradation_with_params(
    image: Image.Image,
    degradation_type: str,
    severity: str,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, Any]]:
    degradation_type = normalize_degradation_type(degradation_type)
    severity = normalize_severity(severity)
    width, height = image.size
    if degradation_type == "gaussian_blur":
        radius_map = {"low": 1.2, "medium": 2.4, "high": 4.0}
        radius = radius_map.get(severity, 2.4)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": radius}
    if degradation_type == "speckle_noise":
        sigma_map = {"low": 0.05, "medium": 0.12, "high": 0.2}
        sigma = sigma_map.get(severity, 0.12)
        noise_seed = rng.randint(0, 10_000_000)
        array = np.asarray(image).astype(np.float32) / 255.0
        noise = np.random.default_rng(noise_seed).normal(0.0, sigma, array.shape)
        speckled = np.clip(array + array * noise, 0.0, 1.0)
        return Image.fromarray((speckled * 255).astype(np.uint8)), {"sigma": sigma, "noise_seed": noise_seed}
    if degradation_type == "resolution_drop":
        scale_map = {"low": 0.8, "medium": 0.55, "high": 0.35}
        scale = scale_map.get(severity, 0.55)
        downsampled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        downsampled = image.resize(downsampled_size, Image.Resampling.BILINEAR)
        restored = downsampled.resize((width, height), Image.Resampling.BICUBIC)
        return restored, {"scale": scale, "downsampled_size": list(downsampled_size)}
    if degradation_type == "contrast_brightness_shift":
        span_map = {"low": 0.12, "medium": 0.22, "high": 0.34}
        span = span_map.get(severity, 0.22)
        contrast_factor = 1.0 + rng.uniform(-span, span)
        brightness_factor = 1.0 + rng.uniform(-span, span)
        degraded = ImageEnhance.Contrast(image).enhance(contrast_factor)
        degraded = ImageEnhance.Brightness(degraded).enhance(brightness_factor)
        return degraded, {
            "span": span,
            "contrast_factor": contrast_factor,
            "brightness_factor": brightness_factor,
        }
    if degradation_type == "random_crop":
        crop_ratio_map = {"low": 0.92, "medium": 0.8, "high": 0.65}
        ratio = crop_ratio_map.get(severity, 0.8)
        crop_w = max(1, int(width * ratio))
        crop_h = max(1, int(height * ratio))
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.Resampling.BICUBIC), {
            "crop_ratio": ratio,
            "crop_box": [left, top, left + crop_w, top + crop_h],
            "resized_to": [width, height],
        }
    if degradation_type == "local_occlusion":
        coverage_map = {"low": 0.12, "medium": 0.22, "high": 0.35}
        coverage = coverage_map.get(severity, 0.22)
        occ_w = max(1, int(width * coverage))
        occ_h = max(1, int(height * coverage))
        left = rng.randint(0, max(0, width - occ_w))
        top = rng.randint(0, max(0, height - occ_h))
        occluded = image.copy()
        occluded.paste(Image.new("RGB", (occ_w, occ_h), color=(0, 0, 0)), (left, top))
        return occluded, {
            "coverage": coverage,
            "occlusion_box": [left, top, left + occ_w, top + occ_h],
            "fill": [0, 0, 0],
        }
    if degradation_type == "center_mask":
        ratio_map = {"low": 0.18, "medium": 0.3, "high": 0.42}
        ratio = ratio_map.get(severity, 0.3)
        mask_w = int(width * ratio)
        mask_h = int(height * ratio)
        left = (width - mask_w) // 2
        top = (height - mask_h) // 2
        masked = image.copy()
        masked.paste(Image.new("RGB", (mask_w, mask_h), color=(0, 0, 0)), (left, top))
        return masked, {"mask_ratio": ratio, "mask_box": [left, top, left + mask_w, top + mask_h], "fill": [0, 0, 0]}
    if degradation_type == "border_truncate":
        ratio_map = {"low": 0.08, "medium": 0.15, "high": 0.22}
        ratio = ratio_map.get(severity, 0.15)
        left = int(width * ratio)
        top = int(height * ratio)
        cropped = image.crop((left, top, width, height))
        padded = ImageOps.pad(cropped, (width, height), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
        return padded, {"truncate_ratio": ratio, "crop_box": [left, top, width, height], "padded_to": [width, height]}
    raise ValueError(f"Unsupported degradation_type: {degradation_type}")


def create_augmented_sample(
    sample: dict[str, Any],
    output_dir: str | Path,
    severity: str,
    degradation_type: str,
    rng: random.Random,
) -> dict[str, Any]:
    source_image_path = sample.get("_source_image_path", sample["image_path"])
    image = load_image(source_image_path)
    augmented = apply_degradation(image, degradation_type, severity, rng)
    planned_sample_id = str(sample.get("sample_id") or "")
    planned_image_path = sample.get("image_path")
    if (
        sample.get("is_counterfactual")
        and sample.get("degradation_type") == degradation_type
        and sample.get("severity") == severity
        and planned_sample_id
        and planned_image_path
    ):
        out_path = Path(planned_image_path)
        result_sample_id = planned_sample_id
    else:
        suffix = Path(source_image_path).suffix or ".png"
        out_path = Path(output_dir) / f"{sample['sample_id']}_{degradation_type}_{severity}{suffix}"
        result_sample_id = f"{sample['sample_id']}__{degradation_type}_{severity}"
    saved_path = save_image(augmented, out_path)
    if not Path(saved_path).exists():
        raise RuntimeError(
            "Augmented image was not written to disk: "
            f"sample_id={sample.get('sample_id')} "
            f"degradation_type={degradation_type} "
            f"severity={severity} "
            f"saved_path={saved_path}"
        )
    result = dict(sample)
    result["sample_id"] = result_sample_id
    result["image_path"] = saved_path
    result["is_counterfactual"] = True
    result["degradation_type"] = degradation_type
    result["severity"] = severity
    result["source_sample_id"] = sample.get("source_sample_id", sample["sample_id"])
    return result
