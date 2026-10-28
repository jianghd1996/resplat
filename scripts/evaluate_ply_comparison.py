#!/usr/bin/env python3
"""Render and compare ReSplat and post-optimization Gaussian PLY files."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from gsplat.rendering import rasterization
from plyfile import PlyData


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_colmap import compute_metrics, load_colmap_scene, load_native_images
from src.competition.scoring import (
    score_lpips, score_psnr, score_ssim, score_time_efficiency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two Gaussian PLY files on the exact same native-resolution views."
    )
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--baseline-ply", type=Path, default=None)
    parser.add_argument("--optimized-ply", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--render-chunk-size", type=int, default=4)
    return parser.parse_args()


def _field(vertex, name):
    if name not in vertex.data.dtype.names:
        raise ValueError(f"PLY is missing required property: {name}")
    return np.asarray(vertex[name], dtype=np.float32)


def load_gaussians(path: Path, device: str):
    vertex = PlyData.read(path)["vertex"]
    names = set(vertex.data.dtype.names)
    means = np.stack([_field(vertex, axis) for axis in ("x", "y", "z")], axis=-1)
    scales = np.exp(np.stack([_field(vertex, f"scale_{i}") for i in range(3)], axis=-1))
    opacities = 1.0 / (1.0 + np.exp(-_field(vertex, "opacity")))

    # Standard 3DGS PLY stores rot_0..3 as wxyz. Covariances make the
    # renderer independent of ReSplat's internal quaternion convention.
    q = np.stack([_field(vertex, f"rot_{i}") for i in range(4)], axis=-1)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    qw, qx, qy, qz = (q[:, i] for i in range(4))
    rotations = np.stack([
        1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw),
        2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw),
        2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy),
    ], axis=-1).reshape(-1, 3, 3)
    covariances = rotations @ np.eye(3, dtype=np.float32)[None] * scales[:, None, :] ** 2
    covariances = covariances @ np.transpose(rotations, (0, 2, 1))

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    rest_count = len(rest_names)
    if rest_count % 3:
        raise ValueError(f"Invalid f_rest property count in {path}: {rest_count}")
    d_sh = 1 + rest_count // 3
    degree = int(math.sqrt(d_sh)) - 1
    if (degree + 1) ** 2 != d_sh:
        raise ValueError(f"Invalid SH coefficient count in {path}: {d_sh}")
    harmonics = np.zeros((len(vertex), 3, d_sh), dtype=np.float32)
    harmonics[:, :, 0] = np.stack(
        [_field(vertex, f"f_dc_{i}") for i in range(3)], axis=-1
    )
    if rest_names:
        rest = np.stack([_field(vertex, name) for name in rest_names], axis=-1)
        harmonics[:, :, 1:] = rest.reshape(len(vertex), 3, d_sh - 1)

    tensor = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to(device)
    # gsplat accepts wxyz here; covariances are supplied and define orientation.
    return {
        "means": tensor(means), "quats": tensor(q), "scales": tensor(scales),
        "opacities": tensor(opacities), "colors": tensor(np.transpose(harmonics, (0, 2, 1))),
        "covars": tensor(covariances), "sh_degree": degree,
    }


@torch.no_grad()
def render_ply(gaussians, c2w, intrinsics, image_shape, near, far, chunk_size, device):
    height, width = image_shape
    c2w = torch.from_numpy(np.asarray(c2w, dtype=np.float32)).to(device)
    intrinsics = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).to(device)
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= width
    intrinsics[:, 1] *= height
    outputs = []
    for start in range(0, len(c2w), chunk_size):
        end = min(start + chunk_size, len(c2w))
        colors, _, _ = rasterization(
            means=gaussians["means"], quats=gaussians["quats"],
            scales=gaussians["scales"], opacities=gaussians["opacities"],
            colors=gaussians["colors"], sh_degree=gaussians["sh_degree"],
            viewmats=torch.linalg.inv(c2w[start:end]), Ks=intrinsics[start:end],
            width=width, height=height, near_plane=near, far_plane=far,
            eps2d=0.1, rasterize_mode="antialiased", packed=True,
            absgrad=False, sparse_grad=False, render_mode="RGB",
            covars=gaussians["covars"],
        )
        outputs.append(colors.permute(0, 3, 1, 2))
    return torch.cat(outputs, dim=0)


def save_renders(rendered, names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, name in zip(rendered, names):
        pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pixels).save(output_dir / f"{Path(name).stem}.png")


def quality_score(metrics):
    components = {
        "psnr": round(score_psnr(metrics["psnr"]), 4),
        "ssim": round(score_ssim(metrics["ssim"]), 4),
        "lpips": round(score_lpips(metrics["lpips"]), 4),
    }
    return components, round(sum(components.values()), 4)


def add_full_score(result, warm_seconds):
    if warm_seconds is None:
        result["warm_end_to_end_seconds"] = None
        result["score_total_100"] = None
        return
    time_component = round(score_time_efficiency(float(warm_seconds), 30.0), 4)
    result["warm_end_to_end_seconds"] = round(float(warm_seconds), 6)
    result["time_score_component"] = time_component
    result["score_total_100"] = round(
        result["quality_score_total_80"] + time_component, 4
    )


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    pipeline_output = args.pipeline_output.expanduser().resolve()
    baseline_ply = (args.baseline_ply or pipeline_output / "resplat" / "gaussians.ply").resolve()
    optimized_ply = (args.optimized_ply or pipeline_output / "module_b" / "gaussians.ply").resolve()
    output = (args.output or pipeline_output / "comparison").resolve()
    frame_path = pipeline_output / "resplat" / "reconstruction_frame.json"
    baseline_metrics_path = pipeline_output / "resplat" / "metrics.json"
    for path in (scene, baseline_ply, optimized_ply, frame_path, baseline_metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with frame_path.open(encoding="utf-8") as f:
        frame_info = json.load(f)
    with baseline_metrics_path.open(encoding="utf-8") as f:
        original_metrics_file = json.load(f)
    evaluation_names = original_metrics_file.get("config", {}).get("evaluation_image_names")
    if not evaluation_names:
        raise ValueError("Baseline metrics.json has no fixed evaluation_image_names list")

    scene_data = load_colmap_scene(str(scene), args.sparse_dir, args.images_dir)
    index_by_name = {name: i for i, name in enumerate(scene_data["image_names"])}
    missing = [name for name in evaluation_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Evaluation images are missing from the scene: {missing[:5]}")
    indices = [index_by_name[name] for name in evaluation_names]
    image_paths = [scene_data["image_paths"][i] for i in indices]
    sizes = {Image.open(path).size for path in image_paths}
    if len(sizes) != 1:
        raise ValueError(f"Evaluation images have mixed native resolutions: {sorted(sizes)}")
    width, height = sizes.pop()
    target_images = load_native_images(image_paths)

    reference = np.asarray(frame_info["reference_c2w_original"], dtype=np.float32)
    world_to_model = np.linalg.inv(reference)
    c2w = np.stack([world_to_model @ scene_data["c2w"][i] for i in indices])
    intrinsics = np.stack([scene_data["intrinsics"][i] for i in indices])

    results = {}
    for label, ply_path in (("resplat", baseline_ply), ("optimized", optimized_ply)):
        print(f"\nEvaluating {label}: {ply_path}")
        start = time.perf_counter()
        gaussians = load_gaussians(ply_path, args.device)
        rendered = render_ply(
            gaussians, c2w, intrinsics, (height, width), args.near, args.far,
            args.render_chunk_size, args.device,
        )
        result_dir = output / label
        save_renders(rendered, evaluation_names, result_dir / "rendered")
        metrics = compute_metrics(
            rendered, target_images, evaluation_names, str(result_dir), device=args.device,
        )
        components, quality_total = quality_score(metrics["mean"])
        results[label] = {
            "ply": str(ply_path), "metrics": metrics["mean"],
            "quality_score_components": components,
            "quality_score_total_80": quality_total,
            "render_and_metric_seconds": round(time.perf_counter() - start, 6),
            "rendered_dir": str(result_dir / "rendered"),
        }

    baseline_report_path = pipeline_output / "performance_quality_report.json"
    scene_timing_path = pipeline_output / "resplat" / "scene_timing.json"
    optimizer_report_path = pipeline_output / "module_b" / "reconstruction_full_report.json"
    baseline_warm_seconds = None
    resplat_output_seconds = None
    if baseline_report_path.is_file():
        with baseline_report_path.open(encoding="utf-8") as f:
            baseline_report = json.load(f)
        timing = baseline_report.get("timing", {})
        baseline_warm_seconds = timing.get("warm_end_to_end_seconds")
        resplat_output_seconds = timing.get("data_to_saved_model_outputs_seconds")
    elif scene_timing_path.is_file():
        # A previous one-process Reconstruction_full run may have failed after
        # ReSplat inference but before run_pipeline.py wrote its final report.
        with scene_timing_path.open(encoding="utf-8") as f:
            scene_timing = json.load(f)
        resplat_output_seconds = max(
            0.0,
            float(scene_timing["scene_processing_seconds"])
            - float(scene_timing.get("quality_metric_evaluation_seconds", 0.0)),
        )
        baseline_warm_seconds = resplat_output_seconds
    optimizer_seconds = None
    if optimizer_report_path.is_file():
        with optimizer_report_path.open(encoding="utf-8") as f:
            optimizer_report = json.load(f)
        optimizer_seconds = optimizer_report.get("stage_wall_seconds")

    add_full_score(results["resplat"], baseline_warm_seconds)
    optimized_warm_seconds = None
    if resplat_output_seconds is not None and optimizer_seconds is not None:
        optimized_warm_seconds = float(resplat_output_seconds) + float(optimizer_seconds)
    add_full_score(results["optimized"], optimized_warm_seconds)

    base, opt = results["resplat"]["metrics"], results["optimized"]["metrics"]
    report = {
        "schema_version": 1,
        "scene": str(scene),
        "evaluation_view_count": len(evaluation_names),
        "evaluation_image_names": evaluation_names,
        "resolution": f"{height}x{width}",
        "same_views_and_native_resolution": True,
        "coordinate_frame": frame_info.get("coordinate_frame"),
        "results": results,
        "optimized_minus_resplat": {
            "psnr": round(opt["psnr"] - base["psnr"], 4),
            "ssim": round(opt["ssim"] - base["ssim"], 4),
            "lpips": round(opt["lpips"] - base["lpips"], 4),
            "quality_score_total_80": round(
                results["optimized"]["quality_score_total_80"]
                - results["resplat"]["quality_score_total_80"], 4
            ),
            "warm_end_to_end_seconds": (
                round(optimized_warm_seconds - baseline_warm_seconds, 6)
                if optimized_warm_seconds is not None and baseline_warm_seconds is not None
                else None
            ),
            "score_total_100": (
                round(
                    results["optimized"]["score_total_100"]
                    - results["resplat"]["score_total_100"], 4
                )
                if results["optimized"]["score_total_100"] is not None
                and results["resplat"]["score_total_100"] is not None
                else None
            ),
        },
    }
    report_path = output / "comparison_report.json"
    output.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nComparison complete")
    print(f"  ReSplat:   PSNR {base['psnr']:.3f}, SSIM {base['ssim']:.3f}, LPIPS {base['lpips']:.3f}")
    print(f"  Optimized: PSNR {opt['psnr']:.3f}, SSIM {opt['ssim']:.3f}, LPIPS {opt['lpips']:.3f}")
    print(f"  Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
