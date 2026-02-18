import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX with fallback metadata.")
    parser.add_argument("--onnx", required=True, help="Input ONNX path")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument("--workspace_gb", type=float, default=2.0, help="Workspace size in GB")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 engine build")
    parser.add_argument("--verbose", action="store_true", help="Verbose TensorRT logging")
    parser.add_argument("--allow_fallback", action="store_true", help="Write fallback metadata on build failure")
    parser.add_argument("--meta_out", default=None, help="Optional metadata json output")
    return parser.parse_args()


def detect_tactic_sources():
    try:
        import tensorrt as trt
    except Exception:
        return []

    tactic_names = []
    if hasattr(trt, "TacticSource"):
        for name in dir(trt.TacticSource):
            if name.startswith("_"):
                continue
            tactic_names.append(name)
    return sorted(tactic_names)


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path
    engine_path = Path(args.engine)
    if not engine_path.is_absolute():
        engine_path = PROJECT_ROOT / engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "upsample" / "build_trt_engine.py"),
        "--input",
        str(onnx_path),
        "--output",
        str(engine_path),
        "--workspace_gb",
        str(args.workspace_gb),
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.verbose:
        cmd.append("--verbose")

    started = datetime.utcnow().isoformat() + "Z"
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    success = result.returncode == 0 and engine_path.exists()

    metadata = {
        "started_utc": started,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "workspace_gb": args.workspace_gb,
        "fp16": bool(args.fp16),
        "verbose": bool(args.verbose),
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "tactic_sources_available": detect_tactic_sources(),
    }

    if success:
        metadata["effective_backend"] = "tensorrt"
    elif args.allow_fallback:
        if onnx_path.exists():
            metadata["fallback_backend"] = "onnxruntime_cuda"
        else:
            metadata["fallback_backend"] = "torch"
        metadata["effective_backend"] = metadata["fallback_backend"]

    if args.meta_out:
        meta_path = Path(args.meta_out)
        if not meta_path.is_absolute():
            meta_path = PROJECT_ROOT / meta_path
    else:
        meta_path = engine_path.with_suffix(".build_meta.json")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not success:
        if args.allow_fallback:
            print(json.dumps(metadata, indent=2))
            return
        raise RuntimeError(f"TensorRT build failed. See metadata: {meta_path}")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
