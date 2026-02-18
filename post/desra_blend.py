import json
from pathlib import Path

import cv2
import numpy as np

from .frame_utils import ensure_dir, list_frames, read_rgb, save_rgb


def _artifact_mask(sharp_rgb, clean_rgb, threshold, blur_radius, blend_strength):
    sharp_f = sharp_rgb.astype(np.float32) / 255.0
    clean_f = clean_rgb.astype(np.float32) / 255.0

    diff = np.mean(np.abs(sharp_f - clean_f), axis=2)
    gray = cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge /= (edge.max() + 1e-6)

    raw = diff * edge
    mask = (raw > float(threshold)).astype(np.float32)

    blur_radius = int(blur_radius)
    if blur_radius > 0:
        kernel = max(1, blur_radius * 2 + 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

    mask = np.clip(mask * float(blend_strength), 0.0, 1.0)
    return mask


def _resolve_clean_dir(stage_config, context, sharp_frame_paths):
    clean_dir = stage_config.get("clean_dir")
    if clean_dir:
        clean_dir = Path(clean_dir)
        if clean_dir.exists():
            return clean_dir, "provided_clean_dir"

    clean_upsampler = stage_config.get("clean_upsampler")
    if clean_upsampler and context.get("run_upsampler_on_dir"):
        cache_dir_name = stage_config.get("clean_cache_dir_name", "clean_sr")
        clean_cache_dir = ensure_dir(Path(context["run_dir"]) / cache_dir_name)
        if not list_frames(clean_cache_dir):
            context["run_upsampler_on_dir"](
                upsampler_key=clean_upsampler,
                source_dir=context["base_frames_dir"],
                output_dir=clean_cache_dir,
                max_frames=len(sharp_frame_paths),
            )
        return clean_cache_dir, f"shadow_upsampler:{clean_upsampler}"

    return Path(context["input_dir"]), "sharp_as_clean_fallback"


def run_stage(input_dir, output_dir, stage_config, context):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    mask_dir = ensure_dir(output_dir / "masks")

    sharp_frame_paths = list_frames(input_dir)
    clean_dir, clean_source = _resolve_clean_dir(stage_config, context, sharp_frame_paths)
    clean_frame_map = {frame.name: frame for frame in list_frames(clean_dir)}

    processed = 0
    for sharp_path in sharp_frame_paths:
        clean_path = clean_frame_map.get(sharp_path.name, sharp_path)
        sharp_rgb = read_rgb(sharp_path)
        clean_rgb = read_rgb(clean_path)
        if clean_rgb.shape[:2] != sharp_rgb.shape[:2]:
            clean_rgb = cv2.resize(
                clean_rgb,
                (sharp_rgb.shape[1], sharp_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        mask = _artifact_mask(
            sharp_rgb=sharp_rgb,
            clean_rgb=clean_rgb,
            threshold=stage_config.get("mask_threshold", 0.12),
            blur_radius=stage_config.get("mask_blur_radius", 3),
            blend_strength=stage_config.get("blend_strength", 0.8),
        )

        sharp_f = sharp_rgb.astype(np.float32) / 255.0
        clean_f = clean_rgb.astype(np.float32) / 255.0
        out = mask[..., None] * clean_f + (1.0 - mask[..., None]) * sharp_f
        out_rgb = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
        save_rgb(output_dir / sharp_path.name, out_rgb)

        if stage_config.get("save_mask_preview", True):
            save_rgb(mask_dir / sharp_path.name, np.repeat((mask * 255.0).astype(np.uint8)[..., None], 3, axis=2))

        processed += 1

    metadata = {
        "stage": "desra_blend",
        "processed_frames": processed,
        "clean_source": clean_source,
        "settings": {
            "mask_threshold": stage_config.get("mask_threshold", 0.12),
            "mask_blur_radius": stage_config.get("mask_blur_radius", 3),
            "blend_strength": stage_config.get("blend_strength", 0.8),
        },
    }
    with open(output_dir / "stage_metadata.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    return {"output_dir": str(output_dir), "metadata": metadata}
