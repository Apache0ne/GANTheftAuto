import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create side-by-side montage across A/B runs.")
    parser.add_argument("--run", action="append", required=True, help="Run spec name=frames_dir")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to render")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--target_height", type=int, default=256, help="Tile height")
    return parser.parse_args()


def list_frames(path):
    path = Path(path)
    return sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[..., ::-1]


def make_tile(img, label, target_h):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_LINEAR)
    cv2.putText(resized, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    return resized


def main():
    args = parse_args()
    tiles = []
    for run_spec in args.run:
        if "=" not in run_spec:
            raise ValueError(f"Invalid --run value: {run_spec}. Expected name=frames_dir")
        name, frame_dir = run_spec.split("=", 1)
        frames = list_frames(frame_dir)
        if not frames:
            continue
        idx = min(max(0, args.frame_index), len(frames) - 1)
        rgb = read_rgb(frames[idx])
        tiles.append(make_tile(rgb, name, args.target_height))

    if not tiles:
        raise RuntimeError("No tiles available for montage.")

    montage = np.concatenate(tiles, axis=1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage[..., ::-1])
    print(f"Saved montage: {out_path}")


if __name__ == "__main__":
    main()
