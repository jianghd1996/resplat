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
`MODEL_ZOO.md`. Keep the selected preset and parameters fixed across scenes.

## Resolution and metrics

The selected preset controls the resolution used to prepare network inputs.
After the unchanged ReSplat forward pass, the final Gaussian model is rendered
again at the native pixel dimensions of the source images under `images_4/`.
The saved `resplat/rendered/*.png` files use that native resolution.
PSNR/SSIM/LPIPS compare these final renders against the original source images
at the same native resolution, without resizing either side for metric
calculation. This means low-resolution presets still receive native-resolution
final renders for scoring; their network input remains at the preset size.

The current implementation expects all images in a scene to have a uniform
native resolution and fails explicitly if they do not. For the demo scene,
the expected source/output/metric resolution is 537x954, while the 512x960
preset's network input is resized to 512x896.

## Timing and score definitions

The model is loaded before the warm scene timer starts. The timer then covers
COLMAP/data loading, Module A view selection and preprocessing, ReSplat
inference, final native-resolution rendering, and saving the PLY/PNG outputs.
PSNR/SSIM/LPIPS are offline evaluation metrics; their compute time is recorded
separately and excluded from the project's practical time score. Module B
runtime is included in the end-to-end time.

For the project score, the time component is calculated from total warm
end-to-end time against a fixed 30-second budget. Full time points are awarded
at 10% of budget or less (3 seconds), zero at or above 30 seconds, and values
between are linearly interpolated. The original official Module B constraint
is also reported independently: `Module B < 30000 * T_iter`. Pass
`--t-iter-sec` to check it.

Module A selection and ReSplat run inside the existing inference entry point,
so their times are not separated; the report labels them as a combined stage.
No ReSplat forward computation is changed.

## Outputs

- `resplat/gaussians.ply`: unchanged ReSplat export.
- `gaussians_refined.ply` and `module_b/gaussians.ply`: Module B output
  (identical to input for this baseline).
- `resplat/rendered/*.png`: native-resolution rendered RGB PNGs.
- `resplat/metrics.json`: mean/per-view metrics and the resolutions used.
- `resplat/scene_timing.json`: model loading, warm scene processing,
  quality-metric evaluation, network input resolution, and native render
  resolution.
- `performance_quality_report.json`: timings, resolutions, score, and the
  separate official Module B budget check.

## Requirements

Use the environment already set up for ReSplat inference (README installation,
including PyTorch/CUDA, gsplat, and pointops). The wrapper itself uses only the
Python standard library and repository modules.
