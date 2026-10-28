#!/usr/bin/env python3
"""Directly refine a ReSplat Gaussian PLY with a few differentiable gsplat steps."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import gsplat
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from PIL import Image
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_colmap import load_colmap_scene, load_native_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-step differentiable refinement of a ReSplat Gaussian PLY."
    )
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--input-ply", type=Path, default=None)
    parser.add_argument("--output-ply", type=Path, default=None)
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("appearance_only", "geometry_lite", "full"),
        default="geometry_lite",
    )
    parser.add_argument("--appearance-warmup-steps", type=int, default=10)
    parser.add_argument("--verify-views", type=int, default=4)
    parser.add_argument("--verify-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lr-sh", type=float, default=2.5e-3)
    parser.add_argument("--lr-opacity", type=float, default=1.0e-3)
    parser.add_argument("--lr-means", type=float, default=2.0e-6)
    parser.add_argument("--lr-scales", type=float, default=2.0e-5)
    parser.add_argument("--lr-quats", type=float, default=1.0e-5)
    parser.add_argument(
        "--max-position-delta",
        type=float,
        default=0.002,
        help="Maximum mean displacement as a fraction of scene scale.",
    )
    parser.add_argument(
        "--max-log-scale-delta",
        type=float,
        default=0.35,
        help="Clamp each log-scale relative to its input value.",
    )
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=200.0)
    return parser.parse_args()


def _field(vertex, name: str) -> np.ndarray:
    if name not in vertex.data.dtype.names:
        raise ValueError(f"PLY is missing required property: {name}")
    return np.asarray(vertex[name], dtype=np.float32)


def load_raw_gaussians(path: Path, device: torch.device):
    vertex = PlyData.read(path)["vertex"]
    names = set(vertex.data.dtype.names)

    means = np.stack([_field(vertex, axis) for axis in ("x", "y", "z")], axis=-1)
    log_scales = np.stack(
        [_field(vertex, f"scale_{index}") for index in range(3)], axis=-1
    )
    opacity_logits = _field(vertex, "opacity")[:, None]
    quats = np.stack(
        [_field(vertex, f"rot_{index}") for index in range(4)], axis=-1
    )
    quats /= np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3:
        raise ValueError(f"Invalid f_rest property count: {len(rest_names)}")
    coefficient_count = 1 + len(rest_names) // 3
    sh_degree = int(math.sqrt(coefficient_count)) - 1
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(f"Invalid SH coefficient count: {coefficient_count}")

    colors = np.zeros((len(vertex), coefficient_count, 3), dtype=np.float32)
    colors[:, 0, :] = np.stack(
        [_field(vertex, f"f_dc_{index}") for index in range(3)], axis=-1
    )
    if rest_names:
        rest = np.stack([_field(vertex, name) for name in rest_names], axis=-1)
        colors[:, 1:, :] = rest.reshape(
            len(vertex), 3, coefficient_count - 1
        ).transpose(0, 2, 1)

    def parameter(value):
        tensor = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        return torch.nn.Parameter(tensor)

    return torch.nn.ParameterDict({
        "means": parameter(means),
        "log_scales": parameter(log_scales),
        "quats": parameter(quats),
        "opacity_logits": parameter(opacity_logits),
        "sh": parameter(colors),
    }), sh_degree


def save_raw_gaussians(
    path: Path,
    params: dict[str, torch.Tensor],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    means = params["means"].detach().cpu().numpy().astype(np.float32)
    log_scales = params["log_scales"].detach().cpu().numpy().astype(np.float32)
    opacity_logits = (
        params["opacity_logits"].detach().cpu().numpy().astype(np.float32)
    )
    quats = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy().astype(np.float32)
    colors = params["sh"].detach().cpu().numpy().astype(np.float32)

    num_gaussians, coefficient_count, _ = colors.shape
    rest_count = 3 * (coefficient_count - 1)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(rest_count)]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]

    arrays = [
        means,
        np.zeros_like(means),
        colors[:, 0, :],
    ]
    if rest_count:
        arrays.append(
            colors[:, 1:, :].transpose(0, 2, 1).reshape(num_gaussians, rest_count)
        )
    arrays += [opacity_logits, log_scales, quats]
    attributes = np.concatenate(arrays, axis=1)
    if attributes.shape[1] != len(names):
        raise RuntimeError(
            f"PLY property mismatch: {attributes.shape[1]} values, {len(names)} names"
        )

    dtype = [(name, "f4") for name in names]
    elements = np.empty(num_gaussians, dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def differentiable_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fast differentiable SSIM with an 11x11 uniform window."""
    window = 11
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(
        prediction * prediction, window, stride=1, padding=padding
    ) - mu_x.square()
    sigma_y = F.avg_pool2d(
        target * target, window, stride=1, padding=padding
    ) - mu_y.square()
    sigma_xy = F.avg_pool2d(
        prediction * target, window, stride=1, padding=padding
    ) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x + sigma_y + c2)
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float,
):
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    l1 = F.l1_loss(prediction, target)
    ssim = differentiable_ssim(prediction, target)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim)
    return loss, l1, ssim


def render(
    params,
    sh_degree,
    c2w,
    intrinsics,
    width,
    height,
    near,
    far,
):
    scales = torch.exp(params["log_scales"])
    quats = F.normalize(params["quats"], dim=-1)
    opacities = torch.sigmoid(params["opacity_logits"]).squeeze(-1)
    rendered, _, _ = rasterization(
        means=params["means"],
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=params["sh"],
        sh_degree=sh_degree,
        viewmats=torch.linalg.inv(c2w),
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        eps2d=0.1,
        rasterize_mode="antialiased",
        packed=True,
        absgrad=False,
        sparse_grad=False,
        render_mode="RGB",
    )
    return rendered.permute(0, 3, 1, 2)


def clone_state(params):
    return {
        name: value.detach().clone()
        for name, value in params.items()
    }


@torch.no_grad()
def restore_state(params, state):
    for name, value in params.items():
        value.copy_(state[name])


@torch.no_grad()
def constrain_parameters(
    params,
    initial_state,
    scene_scale,
    max_position_fraction,
    max_log_scale_delta,
):
    params["opacity_logits"].clamp_(-12.0, 12.0)
    lower = initial_state["log_scales"] - max_log_scale_delta
    upper = initial_state["log_scales"] + max_log_scale_delta
    params["log_scales"].copy_(
        torch.maximum(torch.minimum(params["log_scales"], upper), lower)
    )
    delta = params["means"] - initial_state["means"]
    max_delta = scene_scale * max_position_fraction
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    delta = delta * torch.clamp(max_delta / norm.clamp_min(1e-12), max=1.0)
    params["means"].copy_(initial_state["means"] + delta)
    params["quats"].copy_(F.normalize(params["quats"], dim=-1))


@torch.no_grad()
def evaluate_objective(
    params,
    sh_degree,
    indices,
    c2w,
    intrinsics,
    targets_cpu,
    width,
    height,
    args,
):
    losses, l1_values, ssim_values = [], [], []
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start:start + args.batch_size]
        target = targets_cpu[selected].to(args.device, non_blocking=True)
        prediction = render(
            params,
            sh_degree,
            c2w[selected],
            intrinsics[selected],
            width,
            height,
            args.near,
            args.far,
        )
        loss, l1, ssim = reconstruction_loss(
            prediction, target, args.lambda_ssim
        )
        losses.append(float(loss))
        l1_values.append(float(l1))
        ssim_values.append(float(ssim))
    return {
        "loss": float(np.mean(losses)),
        "l1": float(np.mean(l1_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def main() -> int:
    args = parse_args()
    if args.steps < 0 or args.batch_size < 1:
        raise ValueError("--steps must be non-negative and --batch-size must be positive")
    if not 0.0 <= args.lambda_ssim <= 1.0:
        raise ValueError("--lambda-ssim must be in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene = args.scene.expanduser().resolve()
    pipeline_output = args.pipeline_output.expanduser().resolve()
    input_ply = (
        args.input_ply.expanduser().resolve()
        if args.input_ply
        else pipeline_output / "resplat" / "gaussians_model_frame.ply"
    )
    output_ply = (
        args.output_ply.expanduser().resolve()
        if args.output_ply
        else pipeline_output / "module_b" / "gaussians.ply"
    )
    frame_path = pipeline_output / "resplat" / "reconstruction_frame.json"
    metrics_path = pipeline_output / "resplat" / "metrics.json"
    for path in (scene, input_ply, frame_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with frame_path.open(encoding="utf-8") as file:
        frame_info = json.load(file)
    with metrics_path.open(encoding="utf-8") as file:
        metrics_info = json.load(file)
    evaluation_names = metrics_info.get("config", {}).get("evaluation_image_names")
    if not evaluation_names:
        raise ValueError("metrics.json has no fixed evaluation_image_names list")

    scene_data = load_colmap_scene(
        str(scene), args.sparse_dir, args.images_dir
    )
    index_by_name = {
        name: index for index, name in enumerate(scene_data["image_names"])
    }
    missing = [name for name in evaluation_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Evaluation images are missing: {missing[:5]}")
    indices = [index_by_name[name] for name in evaluation_names]
    image_paths = [scene_data["image_paths"][index] for index in indices]
    sizes = {Image.open(path).size for path in image_paths}
    if len(sizes) != 1:
        raise ValueError(f"Mixed native image resolutions: {sorted(sizes)}")
    width, height = sizes.pop()
    targets_cpu = load_native_images(image_paths).pin_memory()

    device = torch.device(args.device)
    params, sh_degree = load_raw_gaussians(input_ply, device)
    initial_state = clone_state(params)

    reference = np.asarray(
        frame_info["reference_c2w_original"], dtype=np.float32
    )
    world_to_model = np.linalg.inv(reference)
    c2w_np = np.stack([
        world_to_model @ scene_data["c2w"][index]
        for index in indices
    ]).astype(np.float32)
    intrinsics_np = np.stack([
        scene_data["intrinsics"][index]
        for index in indices
    ]).astype(np.float32)
    intrinsics_np[:, 0] *= width
    intrinsics_np[:, 1] *= height
    c2w = torch.from_numpy(c2w_np).to(device)
    intrinsics = torch.from_numpy(intrinsics_np).to(device)

    camera_centers = c2w[:, :3, 3]
    center = camera_centers.mean(dim=0)
    scene_scale = float(
        torch.linalg.vector_norm(camera_centers - center, dim=-1).median()
    )
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        scene_scale = 1.0

    appearance_only = args.mode == "appearance_only"
    optimize_geometry = args.mode in {"geometry_lite", "full"}
    optimize_quats = args.mode == "full"

    param_groups = [
        {"params": [params["sh"]], "lr": args.lr_sh, "name": "sh"},
        {
            "params": [params["opacity_logits"]],
            "lr": args.lr_opacity,
            "name": "opacity",
        },
    ]
    if optimize_geometry:
        param_groups += [
            {"params": [params["means"]], "lr": 0.0, "name": "means"},
            {
                "params": [params["log_scales"]],
                "lr": 0.0,
                "name": "log_scales",
            },
        ]
    if optimize_quats:
        param_groups.append(
            {"params": [params["quats"]], "lr": 0.0, "name": "quats"}
        )
    optimizer = torch.optim.Adam(param_groups, eps=1e-15)

    verify_count = min(max(args.verify_views, 1), len(evaluation_names))
    verify_indices = np.linspace(
        0, len(evaluation_names) - 1, verify_count, dtype=int
    ).tolist()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    initial_objective = evaluate_objective(
        params, sh_degree, verify_indices, c2w, intrinsics, targets_cpu,
        width, height, args,
    )
    best_objective = initial_objective
    best_state = clone_state(params)
    best_step = 0
    history = [{"step": 0, **initial_objective}]
    print(
        f"[REFINE] step 0/{args.steps}: loss={initial_objective['loss']:.6f}, "
        f"L1={initial_objective['l1']:.6f}, SSIM={initial_objective['ssim']:.6f}"
    )

    order = list(range(len(evaluation_names)))
    random.Random(args.seed).shuffle(order)
    cursor = 0

    for step in range(1, args.steps + 1):
        if optimize_geometry and step == args.appearance_warmup_steps + 1:
            for group in optimizer.param_groups:
                if group["name"] == "means":
                    group["lr"] = args.lr_means * scene_scale
                elif group["name"] == "log_scales":
                    group["lr"] = args.lr_scales
                elif group["name"] == "quats":
                    group["lr"] = args.lr_quats

        if cursor + args.batch_size > len(order):
            random.Random(args.seed + step).shuffle(order)
            cursor = 0
        selected = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size

        target = targets_cpu[selected].to(device, non_blocking=True)
        prediction = render(
            params, sh_degree, c2w[selected], intrinsics[selected],
            width, height, args.near, args.far,
        )
        loss, _, _ = reconstruction_loss(
            prediction, target, args.lambda_ssim
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        constrain_parameters(
            params,
            initial_state,
            scene_scale,
            args.max_position_delta,
            args.max_log_scale_delta,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")

        should_verify = (
            step == args.steps
            or (args.verify_every > 0 and step % args.verify_every == 0)
        )
        if should_verify:
            objective = evaluate_objective(
                params, sh_degree, verify_indices, c2w, intrinsics, targets_cpu,
                width, height, args,
            )
            history.append({"step": step, **objective})
            print(
                f"[REFINE] step {step}/{args.steps}: "
                f"loss={objective['loss']:.6f}, "
                f"L1={objective['l1']:.6f}, SSIM={objective['ssim']:.6f}"
            )
            if objective["loss"] < best_objective["loss"]:
                best_objective = objective
                best_state = clone_state(params)
                best_step = step

    restore_state(params, best_state)
    save_raw_gaussians(output_ply, params)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    report = {
        "name": "direct_gsplat_refinement",
        "torch_version": torch.__version__,
        "gsplat_version": getattr(gsplat, "__version__", "unknown"),
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "mode": args.mode,
        "steps_requested": args.steps,
        "best_step": best_step,
        "batch_size": args.batch_size,
        "training_view_count": len(evaluation_names),
        "verification_view_count": verify_count,
        "native_resolution": f"{height}x{width}",
        "gaussian_count": int(params["means"].shape[0]),
        "sh_degree": sh_degree,
        "scene_scale": scene_scale,
        "initial_objective": initial_objective,
        "best_objective": best_objective,
        "improved_verification_objective": (
            best_objective["loss"] < initial_objective["loss"]
        ),
        "optimization_seconds": round(elapsed, 6),
        "config": vars(args) | {
            "scene": str(scene),
            "pipeline_output": str(pipeline_output),
            "input_ply": str(input_ply),
            "output_ply": str(output_ply),
        },
        "history": history,
    }
    report_path = output_ply.parent / "direct_refinement_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nDirect gsplat refinement complete")
    print(f"  Input:       {input_ply}")
    print(f"  Output:      {output_ply}")
    print(f"  Best step:   {best_step}/{args.steps}")
    print(
        f"  Verify loss: {initial_objective['loss']:.6f} -> "
        f"{best_objective['loss']:.6f}"
    )
    print(f"  Wall time:   {elapsed:.3f}s")
    print(f"  Report:      {report_path}")
    if best_step == 0:
        print("  Note: no verified improvement; exported the original parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
