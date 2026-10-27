# DL3DV competition pipeline (baseline)

This adds an external one-command wrapper around the repository's existing
COLMAP inference path. It does not edit ReSplat's network, checkpoint, or
forward pass. The first Module B adapter is an **identity baseline**: it copies
the generated PLY and verifies that the bytes are unchanged. It is a harness
for local measurements, not a quality-improving method.

## Official demo

Download and extract the DL3DV COLMAP demo linked from the upstream README into
the repository's `datasets/dl3dv-colmap-demo` directory. The selected scene
folder must contain `images_4/` and `sparse/0/` (COLMAP binary or text files).
The default preset expects the 8-view 512x960 checkpoint:

```bash
python run_pipeline.py \
  --scene datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be \
  --output results/competition/demo \
  --t-iter-sec 0.01
```

Replace `0.01` with the measured seconds per iteration of classic 3DGS on the
same machine. If omitted, inference and quality metrics are still produced,
but time-efficiency and total competition scores are left unscored.

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

## Outputs

- `resplat/gaussians.ply`: unchanged ReSplat export.
- `gaussians_refined.ply` and `module_b/gaussians.ply`: Module B output
  (identical to input for this baseline).
- `resplat/rendered/*.png`: rendered RGB PNGs for all non-context views.
- `resplat/metrics.json`: mean and per-view PSNR, SSIM, and LPIPS.
- `performance_quality_report.json`: timing, budget check, component scores,
  and total score when all required inputs are available.

The score maps PSNR 15–40 dB to 0–20 points, SSIM 0.70–0.95 to 0–20,
LPIPS 0.25–0.05 to 0–40, and Module B runtime to 0–20 (full time points at
10% or less of the 30,000-iteration budget; zero at or above the budget).
Values outside the metric ranges are clamped at the range endpoints.

## Timing interpretation

The report measures ReSplat inference separately and counts only the Module B
adapter against the 30,000-iteration allowance. For the baseline this is
copy-and-verify time. A future optimization must replace the adapter while
preserving its interface and must be evaluated with the same scene-independent
parameters. The time check uses the strict condition `module_b_seconds <
30000 * t_iter_sec`.

## Requirements

Use the environment already set up for ReSplat inference (README installation,
including PyTorch/CUDA, gsplat, and pointops). The wrapper itself uses only the
Python standard library and repository modules.
