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

For example: `--model-preset dl3dv_16v_256x448`. Keep the chosen preset and
all other parameters fixed across evaluation scenes; do not tune by scene.

## Timing definitions

The model is loaded before the warm scene timer starts. The timer then covers
COLMAP/data loading, Module A view selection and preprocessing, ReSplat
inference, rendering, and saving the PLY/PNG outputs. PSNR/SSIM/LPIPS are
offline evaluation metrics; their compute time is recorded separately and is
excluded from the 30-second practical target. The report also preserves the
cold subprocess wall time, which includes model load.

Module A selection and ReSplat run inside the existing upstream inference
entry point, so their times are not separated; the report labels them as a
combined stage. No ReSplat forward-path instrumentation or computation changes
are made.

The **30-second end-to-end target is a practical engineering target**, not a
replacement for the official Module B limit. The official score maps PSNR
15–40 dB to 0–20 points, SSIM 0.70–0.95 to 0–20, LPIPS 0.25–0.05 to 0–40,
and Module B runtime to 0–20 (full time points at 10% or less of the
30,000-iteration budget; zero at or above the budget). Values outside metric
ranges are clamped. Pass `--t-iter-sec` with measured classic 3DGS seconds per
iteration to calculate the official time component and total score.

## Outputs

- `resplat/gaussians.ply`: unchanged ReSplat export.
- `gaussians_refined.ply` and `module_b/gaussians.ply`: Module B output
  (identical to input for this baseline).
- `resplat/rendered/*.png`: rendered RGB PNGs for all non-context views.
- `resplat/metrics.json`: mean and per-view PSNR, SSIM, and LPIPS.
- `resplat/scene_timing.json`: model loading, warm scene processing, and
  quality-metric evaluation timings.
- `performance_quality_report.json`: timings, budget checks, and scores.

The official Module B check is the strict condition
`module_b_seconds < 30000 * t_iter_sec`. The separate practical target is
`warm_end_to_end_seconds <= 30`.

## Requirements

Use the environment already set up for ReSplat inference (README installation,
including PyTorch/CUDA, gsplat, and pointops). The wrapper itself uses only the
Python standard library and repository modules.
