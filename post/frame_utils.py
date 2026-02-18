import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_frames(frame_dir):
    frame_dir = Path(frame_dir)
    return sorted(
        [
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img[..., ::-1]


def save_rgb(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(img)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out[..., ::-1])


def copy_frames(input_dir, output_dir, max_frames=None):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    frame_paths = list_frames(input_dir)
    if max_frames is not None:
        frame_paths = frame_paths[: max_frames]
    for src in frame_paths:
        shutil.copy2(src, output_dir / src.name)
    return [output_dir / frame.name for frame in frame_paths]


def temporal_ema(input_dir, output_dir, alpha=0.65, max_frames=None):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    frame_paths = list_frames(input_dir)
    if max_frames is not None:
        frame_paths = frame_paths[: max_frames]

    ema = None
    output_paths = []
    for frame_path in frame_paths:
        rgb = read_rgb(frame_path).astype(np.float32)
        if ema is None:
            ema = rgb
        else:
            ema = alpha * ema + (1.0 - alpha) * rgb
        out = np.clip(ema, 0, 255).astype(np.uint8)
        out_path = output_dir / frame_path.name
        save_rgb(out_path, out)
        output_paths.append(out_path)
    return output_paths
