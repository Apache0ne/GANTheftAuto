import json
import shlex
import subprocess
from pathlib import Path

from .frame_utils import copy_frames, ensure_dir, list_frames


def run_stage(input_dir, output_dir, stage_config, context):
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)

    if not stage_config.get("offline_only", True):
        raise RuntimeError("osediff_refine must remain offline_only=True.")

    sample_frames = int(stage_config.get("sample_frames", 8))
    sampled_dir = ensure_dir(output_dir / "sampled_input")
    sampled_frames = copy_frames(input_dir=input_dir, output_dir=sampled_dir, max_frames=sample_frames)

    mode = "copy_fallback"
    command_error = None
    command_template = stage_config.get("command_template")
    if command_template:
        cmd = command_template.format(
            input_dir=str(sampled_dir),
            output_dir=str(output_dir),
        )
        try:
            completed = subprocess.run(
                shlex.split(cmd, posix=False),
                check=True,
                capture_output=True,
                text=True,
            )
            mode = "external_command"
            if completed.stdout:
                (output_dir / "command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            if completed.stderr:
                (output_dir / "command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        except Exception as err:
            command_error = str(err)

    if mode != "external_command" and stage_config.get("copy_if_unavailable", True):
        copy_frames(sampled_dir, output_dir, max_frames=sample_frames)

    metadata = {
        "stage": "osediff_refine",
        "mode": mode,
        "offline_only": True,
        "sample_frames": sample_frames,
        "processed_frames": len(list_frames(output_dir)),
        "command_template": command_template,
        "command_error": command_error,
    }
    with open(output_dir / "stage_metadata.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    return {"output_dir": str(output_dir), "metadata": metadata}
