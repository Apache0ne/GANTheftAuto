from .vsr_common import run_vsr_stage


def run_stage(input_dir, output_dir, stage_config, context):
    return run_vsr_stage(
        stage_name="vsr_realbasicvsr",
        input_dir=input_dir,
        output_dir=output_dir,
        stage_config=stage_config,
    )
