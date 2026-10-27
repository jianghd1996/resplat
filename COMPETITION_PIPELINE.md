# DL3DV competition pipeline (baseline)

This adds an external one-command wrapper around the repository's existing
COLMAP inference path. It does not change ReSplat's network, checkpoint, or
forward computation. The first Module B adapter is an **identity baseline**:
it copies the generated PLY and verifies that the bytes are unchanged. It is
a harness for local measurements, not a quality-improving method.

## Official demo

Download and extract the DL3DV COLMAP demo linked from the upstream README.
The selected scene folder must contain `images_4/` and `sparse/0/` (COLMAP
binary or text files). The default preset expects the 8-view 512x960 checkpoint:

```bash
python run_pipeline.py \
  --scene datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be \
  --output results/competition/demo
```

Model weights are expected under `pretrained/` as described in
`MODEL_ZOO.md`. To use the downloaded view-count/resolution variants, select
the matching preset:

- `dl3dv_8v_512x960` (default)
- `dl3dv_8v_256x448`
- `dl3dv_16v_256x448`
- `dl3dv_32v_256x448`
- `dl3dv_8v_256x448_large`

Keep the chosen preset and all other parameters fixed across evaluation scenes.

## Timing and score definitions

The model is loaded before the warm scene timer starts. The timer then covers
COLMAP/data loading, Module A view selection and preprocessing, ReSplat
inference, rendering, and saving the PLY/PNG outputs. PSNR/SSIM/LPIPS are
offline evaluation metrics; their compute time is recorded separately and is
excluded from the project's practical time score. Module B runtime is included
in the end-to-end time.

For the project score, the time component is calculated from total warm
end-to-end time against a fixed 30-second budget. It follows the stated linear
rule: full time points at 10% of budget or less (3 seconds), zero at or above
30 seconds, with linear interpolation in between. This project score basis
reflects the user's chosen practical target.

The original official Module B constraint is also reported independently:
`Module B < 30000 * T_iter`. Pass `--t-iter-sec` to check it. This parameter
is no longer required to calculate the 30-second total-time score.

Module A selection and ReSplat run inside the existing inference entry point,
so their times are not separated; the report labels them as a combined stage.
No ReSplat forward computation is changed.

## Outputs

- `resplat/gaussians.ply`: unchanged ReSplat export.
- `gaussians_refined.ply` and `module_b/gaussians.ply`: Module B output
  (identical to input for this baseline).
- `resplat/rendered/*.png`: rendered RGB PNGs for all non-context views.
- `resplat/metrics.json`: mean and per-view PSNR, SSIM, and LPIPS.
- `resplat/scene_timing.json`: model loading, warm scene processing, and
  quality-metric evaluation timings.
- `performance_quality_report.json`: timings, the 30-second score, and the
  separate official Module B budget check.

## Requirements

Use the environment already set up for ReSplat inference (README installation,
including PyTorch/CUDA, gsplat, and pointops). The wrapper itself uses only the
Python standard library and repository modules.
