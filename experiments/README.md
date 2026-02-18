# SR Lab Pack (Non-Invasive Add-ons)

This folder documents optional quality experiments around the existing GANTheftAuto pipeline.
Baseline behavior is unchanged unless you explicitly run the new tools.

## What was added

- `configs/sr_switchboard.yaml`: upsampler + post-stage switchboard.
- `tools/run_pipeline.py`: outer runner for offline frame/image tests.
- `post/desra_blend.py`: artifact-aware blend stage (opt-in).
- `post/vsr_realbasicvsr.py`: optional temporal VSR wrapper.
- `post/vsr_basicvsrpp.py`: optional temporal VSR wrapper.
- `post/osediff_refine.py`: offline-only refiner wrapper.
- `tools/export_onnx.py`: ONNX export wrapper for SR models.
- `tools/build_trt.py`: TRT build wrapper with fallback metadata.
- `scripts/ab_test.ps1`: A/B benchmark harness.

## Baseline remains unchanged

Keep using existing scripts exactly as before:

```bat
.\scripts\gtav_inference_demo_no_upsample.bat
.\scripts\gtav_inference_demo_tensorrt.bat
```

No core script is required for these experiments.

## Prerequisites

Use your current env (`fashn_parser`) and install optional deps:

```bat
pip install -r requirements-upsample-trt.txt --extra-index-url https://pypi.nvidia.com
pip install pyyaml
```

Model downloads are intentionally not automated.

## Run one experiment manually

Single input image, selected upsampler, one post-stage:

```bat
python tools/run_pipeline.py ^
  --switchboard configs/sr_switchboard.yaml ^
  --upsampler swinir_realsr_x2_trt ^
  --post desra_blend ^
  --input ./data/gtav/2.png ^
  --repeat_input 12 ^
  --out_dir ./outputs/run_manual_test
```

Key outputs:

- `metrics.json`
- `preview_side_by_side.png`
- `upsampled_<upsampler>/`
- `post_XX_<stage>/` (if enabled)

## Run A/B harness

```bat
powershell -ExecutionPolicy Bypass -File .\scripts\ab_test.ps1 -BuildMontage
```

Outputs:

- `./outputs/ab_YYYYMMDD_HHMMSS/ab_results.csv`
- `./outputs/ab_YYYYMMDD_HHMMSS/ab_results.json`
- `./outputs/ab_YYYYMMDD_HHMMSS/ab_montage_frame_XXXX.png` (one per sampled frame when montage is enabled)

## Notes on post stages

- `desra_blend`:
  - Uses artifact-aware mask between sharp and clean SR outputs.
  - Can trigger a shadow clean SR run via `clean_upsampler`.
- `vsr_realbasicvsr` / `vsr_basicvsrpp`:
  - If external VSR command is not configured, falls back to temporal EMA smoothing.
  - Use `smoke_test_frames` for short verification runs.
- `osediff_refine`:
  - Offline-only by design.
  - Runs on a sampled subset of frames.

## Compare template

Use `experiments/results_template.md` to record quality + speed comparisons.
