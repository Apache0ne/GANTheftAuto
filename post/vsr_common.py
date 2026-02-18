import json
import shlex
import subprocess
from pathlib import Path

from .frame_utils import copy_frames, ensure_dir, temporal_ema


def run_vsr_stage(stage_name, input_dir, output_dir, stage_config):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    smoke_frames = stage_config.get("smoke_test_frames")
    if smoke_frames is not None:
        smoke_frames = int(smoke_frames)

    mode = "copy"
    command_template = stage_config.get("command_template")
    model_path = stage_config.get("model_path")
    command_error = None

    if command_template and model_path:
        cmd = command_template.format(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            model_path=str(model_path),
            max_frames=str(smoke_frames if smoke_frames is not None else ""),
            use_fp16=str(bool(stage_config.get("use_fp16", False))).lower(),
            tile=str(stage_config.get("tile", 0)),
        )
        try:
            completed = subprocess.run(
                shlex.split(cmd, posix=False),
                check=True,
                capture_output=True,
                text=True,
            )
            mode = "external_command"
            command_error = None
            if completed.stdout:
                (output_dir / "command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            if completed.stderr:
                (output_dir / "command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        except Exception as err:
            command_error = str(err)

    if mode != "external_command":
        if stage_config.get("fallback_temporal_ema", True):
            temporal_ema(
                input_dir=input_dir,
                output_dir=output_dir,
                alpha=float(stage_config.get("ema_alpha", 0.65)),
                max_frames=smoke_frames,
            )
            mode = "temporal_ema_fallback"
        else:
            copy_frames(input_dir=input_dir, output_dir=output_dir, max_frames=smoke_frames)
            mode = "copy_fallback"

    metadata = {
        "stage": stage_name,
        "mode": mode,
        "smoke_test_frames": smoke_frames,
        "model_path": model_path,
        "command_template": command_template,
        "command_error": command_error,
    }
    with open(output_dir / "stage_metadata.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    return {"output_dir": str(output_dir), "metadata": metadata}
