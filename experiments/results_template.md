# SR Experiment Results Template

Date:
Branch:
GPU:
Driver/CUDA:
Conda env:

## Dataset/Input

- Input source:
- Frame count:
- Resolution:

## Runs

| Run Name | Upsampler | Post Stages | Avg ms/frame | Total sec | GPU mem snapshot (MB) | Notes |
|---|---|---|---:|---:|---:|---|
| baseline_legacy_trt | legacy_trt | none |  |  |  |  |
| legacy_trt_desra | legacy_trt | desra_blend |  |  |  |  |
| legacy_trt_vsr | legacy_trt | vsr_realbasicvsr |  |  |  |  |
| swinir_x2_trt | swinir_realsr_x2_trt | none |  |  |  |  |
| swinir_x2_trt_desra | swinir_realsr_x2_trt | desra_blend |  |  |  |  |
| hat_x2_trt | hat_x2_trt | none |  |  |  |  |
| rgt_x2_trt | rgt_x2_trt | none |  |  |  |  |

## Visual Checks

- Montage path:
- Best texture detail:
- Lowest flicker:
- Least artifacts:
- Preferred tradeoff:

## Decisions

- Keep default upsampler:
- Enable optional post stage(s):
- Re-run needed with different settings:
