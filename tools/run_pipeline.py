import argparse
import copy
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post.frame_utils import ensure_dir, list_frames, read_rgb, save_rgb  # noqa: E402

try:
    import yaml
except ImportError as err:  # pragma: no cover
    raise RuntimeError("PyYAML is required for tools/run_pipeline.py. Install with: pip install pyyaml") from err


class EngineLikeUpsampler:
    def __init__(self, model_path):
        from upsample import upsample as upsample_backend

        self.backend = upsample_backend
        self.model_path = str(model_path)
        self.backend.load(self.model_path)

    def infer(self, image_rgb_norm):
        return self.backend.inference(image_rgb_norm)


class TorchSpandrelUpsampler:
    def __init__(self, weights_path, device=None):
        import torch
        from spandrel import ModelLoader

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.torch = torch
        descriptor = ModelLoader().load_from_file(str(weights_path)).eval().to(self.device)
        self.model = descriptor.model
        self.scale = descriptor.scale
        self.input_channels = descriptor.input_channels
        self.output_channels = descriptor.output_channels

    def infer(self, image_rgb_norm):
        x = self.torch.from_numpy(image_rgb_norm.astype(np.float32))
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x + 1.0) * 0.5
        with self.torch.inference_mode():
            y = self.model(x)
        y = self.torch.clamp(y, 0.0, 1.0)
        y = (y * 255.0).round().to(self.torch.uint8)
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return y


def parse_args():
    parser = argparse.ArgumentParser(description="Non-invasive SR lab pipeline runner.")
    parser.add_argument("--switchboard", default="configs/sr_switchboard.yaml", help="Switchboard YAML path")
    parser.add_argument("--upsampler", default=None, help="Upsampler key from switchboard")
    parser.add_argument("--post", action="append", default=None, help="Post stage key (repeatable)")
    parser.add_argument("--input", default=None, help="Single input frame path")
    parser.add_argument("--frames_dir", default=None, help="Directory of input frames")
    parser.add_argument("--repeat_input", type=int, default=1, help="Repeat single input frame N times")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--make_preview_count", type=int, default=4, help="Create side-by-side preview with N frames")
    parser.add_argument("--live_script", default=None, help="Optional existing script to run before offline processing")
    parser.add_argument("--live_timeout_sec", type=int, default=None, help="Optional timeout for live script run")
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_switchboard(path_value):
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Switchboard not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return path, data


def maybe_run_live_script(args):
    if not args.live_script:
        return
    command = args.live_script
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        shell=True,
        check=True,
        timeout=args.live_timeout_sec,
    )


def prepare_input_frames(args, run_dir):
    input_frames_dir = ensure_dir(run_dir / "input_frames")
    frame_paths = []
    if args.frames_dir:
        source_dir = resolve_path(args.frames_dir)
        source_frames = list_frames(source_dir)
        if args.max_frames is not None:
            source_frames = source_frames[: args.max_frames]
        for index, src in enumerate(source_frames):
            dst = input_frames_dir / f"frame_{index:06d}{src.suffix.lower()}"
            dst.write_bytes(src.read_bytes())
            frame_paths.append(dst)
        return input_frames_dir, frame_paths

    if args.input:
        src = resolve_path(args.input)
        if not src.exists():
            raise FileNotFoundError(f"Input frame not found: {src}")
        repeat_count = max(1, int(args.repeat_input))
        if args.max_frames is not None:
            repeat_count = min(repeat_count, int(args.max_frames))
        for index in range(repeat_count):
            dst = input_frames_dir / f"frame_{index:06d}{src.suffix.lower()}"
            dst.write_bytes(src.read_bytes())
            frame_paths.append(dst)
        return input_frames_dir, frame_paths

    raise ValueError("Provide either --input or --frames_dir for offline processing.")


def create_upsampler(upsampler_cfg):
    upsampler_type = upsampler_cfg.get("type")
    if upsampler_type == "trt_engine":
        model_path = resolve_path(upsampler_cfg["engine_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {model_path}")
        return EngineLikeUpsampler(model_path)
    if upsampler_type == "torch":
        weights_path = resolve_path(upsampler_cfg["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"Torch weights not found: {weights_path}")
        return TorchSpandrelUpsampler(weights_path)
    raise ValueError(f"Unsupported upsampler type: {upsampler_type}")


def extract_primary_output(output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    output = np.asarray(output)
    if output.ndim == 4:
        output = output[0]
    if output.ndim != 3:
        raise ValueError(f"Unexpected upsampler output shape: {output.shape}")
    return output


def to_uint8_rgb(img):
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.min() >= -1.1 and arr.max() <= 1.1:
        arr = (arr + 1.0) * 127.5
    elif arr.min() >= 0.0 and arr.max() <= 1.1:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def run_upsampler_on_dir(source_dir, output_dir, upsampler_key, switchboard, max_frames=None):
    output_dir = ensure_dir(output_dir)
    source_frames = list_frames(source_dir)
    if max_frames is not None:
        source_frames = source_frames[: max_frames]

    upsampler_cfg = switchboard["upsamplers"].get(upsampler_key)
    if upsampler_cfg is None:
        raise KeyError(f"Upsampler key not found in switchboard: {upsampler_key}")

    runner = create_upsampler(upsampler_cfg)

    timings_ms = []
    output_paths = []
    for frame_path in source_frames:
        rgb = read_rgb(frame_path)
        rgb_norm = rgb.astype(np.float32) / 127.5 - 1.0
        start = time.perf_counter()
        out = runner.infer(rgb_norm)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

        primary = extract_primary_output(out)
        primary = to_uint8_rgb(primary)
        out_path = output_dir / frame_path.name
        save_rgb(out_path, primary)
        output_paths.append(out_path)

    metrics = {
        "upsampler_key": upsampler_key,
        "frame_count": len(output_paths),
        "avg_ms_per_frame": float(np.mean(timings_ms)) if timings_ms else 0.0,
        "min_ms_per_frame": float(np.min(timings_ms)) if timings_ms else 0.0,
        "max_ms_per_frame": float(np.max(timings_ms)) if timings_ms else 0.0,
        "total_ms": float(np.sum(timings_ms)) if timings_ms else 0.0,
    }
    return {"output_dir": str(output_dir), "frames": [str(p) for p in output_paths], "metrics": metrics}


def load_stage_runner(stage_name):
    module_map = {
        "desra_blend": "post.desra_blend",
        "vsr_realbasicvsr": "post.vsr_realbasicvsr",
        "vsr_basicvsrpp": "post.vsr_basicvsrpp",
        "osediff_refine": "post.osediff_refine",
    }
    module_path = module_map.get(stage_name)
    if module_path is None:
        raise KeyError(f"Unknown post stage: {stage_name}")
    module = importlib.import_module(module_path)
    return module.run_stage


def create_preview(base_dir, final_dir, out_path, count):
    base_frames = list_frames(base_dir)[:count]
    final_map = {frame.name: frame for frame in list_frames(final_dir)}
    rows = []
    for base_path in base_frames:
        final_path = final_map.get(base_path.name)
        if final_path is None:
            continue
        base = read_rgb(base_path)
        final = read_rgb(final_path)
        target_h = min(256, max(base.shape[0], final.shape[0]))
        base = cv2.resize(base, (int(base.shape[1] * target_h / base.shape[0]), target_h), interpolation=cv2.INTER_NEAREST)
        final = cv2.resize(final, (int(final.shape[1] * target_h / final.shape[0]), target_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.concatenate([base, final], axis=1)
        cv2.putText(canvas, "Base", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Final",
            (base.shape[1] + 8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        rows.append(canvas)

    if rows:
        montage = np.concatenate(rows, axis=0)
        save_rgb(out_path, montage)


def main():
    args = parse_args()
    switchboard_path, switchboard = load_switchboard(args.switchboard)

    maybe_run_live_script(args)
    if not args.input and not args.frames_dir:
        if args.live_script:
            return
        raise ValueError("No offline input provided. Use --input or --frames_dir.")

    out_dir = resolve_path(args.out_dir) if args.out_dir else PROJECT_ROOT / "outputs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = ensure_dir(out_dir)

    upsampler_key = args.upsampler or switchboard.get("default_upsampler")
    if not upsampler_key:
        raise ValueError("No upsampler selected. Set --upsampler or default_upsampler in switchboard.")

    post_stages = copy.deepcopy(args.post if args.post else switchboard.get("default_post_stages", []))

    started = time.perf_counter()
    base_frames_dir, _ = prepare_input_frames(args, out_dir)

    upsampled_dir = out_dir / f"upsampled_{upsampler_key}"
    upsample_result = run_upsampler_on_dir(
        source_dir=base_frames_dir,
        output_dir=upsampled_dir,
        upsampler_key=upsampler_key,
        switchboard=switchboard,
        max_frames=args.max_frames,
    )

    current_dir = Path(upsample_result["output_dir"])
    stage_results = []

    def aux_upsample(upsampler_key, source_dir, output_dir, max_frames=None):
        return run_upsampler_on_dir(
            source_dir=Path(source_dir),
            output_dir=Path(output_dir),
            upsampler_key=upsampler_key,
            switchboard=switchboard,
            max_frames=max_frames,
        )

    for index, stage_name in enumerate(post_stages):
        stage_cfg = copy.deepcopy(switchboard.get("post_stage_configs", {}).get(stage_name, {}))
        stage_out_dir = out_dir / f"post_{index:02d}_{stage_name}"
        context = {
            "project_root": str(PROJECT_ROOT),
            "run_dir": str(out_dir),
            "switchboard_path": str(switchboard_path),
            "switchboard": switchboard,
            "base_frames_dir": str(base_frames_dir),
            "input_dir": str(current_dir),
            "run_upsampler_on_dir": aux_upsample,
        }
        runner = load_stage_runner(stage_name)
        stage_result = runner(
            input_dir=current_dir,
            output_dir=stage_out_dir,
            stage_config=stage_cfg,
            context=context,
        )
        current_dir = Path(stage_result["output_dir"])
        stage_results.append({"stage": stage_name, "result": stage_result})

    if args.make_preview_count and int(args.make_preview_count) > 0:
        create_preview(
            base_dir=base_frames_dir,
            final_dir=current_dir,
            out_path=out_dir / "preview_side_by_side.png",
            count=int(args.make_preview_count),
        )

    total_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "switchboard": str(switchboard_path),
        "upsampler": upsampler_key,
        "post_stages": post_stages,
        "input_frames_dir": str(base_frames_dir),
        "upsample_result": upsample_result,
        "final_frames_dir": str(current_dir),
        "stage_results": stage_results,
        "total_runtime_ms": total_ms,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
