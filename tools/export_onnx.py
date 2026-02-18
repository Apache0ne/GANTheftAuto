import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Export SR models to ONNX (non-invasive wrapper).")
    parser.add_argument("--weights", required=True, help="Path to SR model weights (.pth)")
    parser.add_argument("--onnx", required=True, help="Output ONNX file path")
    parser.add_argument("--input_shape", default="1x48x80x3", help="Static input shape NxHxWxC")
    parser.add_argument("--opset", type=int, default=17, help="Requested ONNX opset")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument("--allow_torch_fallback", action="store_true", help="Write fallback metadata if ONNX export fails")
    parser.add_argument("--meta_out", default=None, help="Optional metadata json output")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "upsample" / "convert_spandrel_pth_to_onnx.py"),
        "--input",
        str(weights),
        "--output",
        str(onnx_path),
        "--input_shape",
        args.input_shape,
        "--opset",
        str(args.opset),
        "--device",
        args.device,
    ]

    started = datetime.utcnow().isoformat() + "Z"
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    success = result.returncode == 0 and onnx_path.exists()

    metadata = {
        "started_utc": started,
        "weights": str(weights),
        "onnx": str(onnx_path),
        "input_shape": args.input_shape,
        "requested_opset": args.opset,
        "device": args.device,
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "recommended_runtime_order": ["tensorrt", "onnxruntime_cuda", "torch"],
    }

    if success:
        metadata["effective_backend"] = "onnx"
    elif args.allow_torch_fallback:
        metadata["fallback_backend"] = "torch"
        metadata["effective_backend"] = "torch"

    if args.meta_out:
        meta_path = Path(args.meta_out)
        if not meta_path.is_absolute():
            meta_path = PROJECT_ROOT / meta_path
    else:
        meta_path = onnx_path.with_suffix(".export_meta.json")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not success:
        if args.allow_torch_fallback:
            print(json.dumps(metadata, indent=2))
            return
        raise RuntimeError(f"ONNX export failed. See metadata: {meta_path}")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
