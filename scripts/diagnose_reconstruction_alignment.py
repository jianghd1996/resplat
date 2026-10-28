#!/usr/bin/env python3
"""Compare Reconstruction_full final cameras with ReSplat's model frame."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_colmap import (
    load_colmap_scene,
    qvec2rotmat,
    read_extrinsics_text,
    read_intrinsics_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the global similarity transform from Reconstruction_full's "
            "final COLMAP cameras to the ReSplat model frame."
        )
    )
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument(
        "--colmap-final", type=Path, default=None,
        help="Folder containing Reconstruction_full cameras.txt and images.txt.",
    )
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def image_c2w(image) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3] = image.tvec
    return np.linalg.inv(w2c)


def fit_similarity(source: np.ndarray, target: np.ndarray):
    """Fit target ~= scale * rotation @ source + translation (Umeyama)."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    source_variance = np.mean(np.sum(source_centered ** 2, axis=1))
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def rotation_error_degrees(reference: np.ndarray, estimate: np.ndarray) -> float:
    relative = reference @ estimate.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalized_intrinsics(camera) -> np.ndarray | None:
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params[:3]
        fy = fx
    else:
        return None
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = fx / camera.width
    matrix[1, 1] = fy / camera.height
    matrix[0, 2] = cx / camera.width
    matrix[1, 2] = cy / camera.height
    return matrix


def find_latest_colmap_final(pipeline_output: Path) -> Path:
    result_root = pipeline_output / "module_b" / "reconstruction_work" / "result_gs"
    candidates = [
        path for path in result_root.glob("*/colmap_final")
        if (path / "cameras.txt").is_file() and (path / "images.txt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No colmap_final with cameras.txt and images.txt found under {result_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    pipeline_output = args.pipeline_output.expanduser().resolve()
    colmap_final = (
        args.colmap_final.expanduser().resolve()
        if args.colmap_final else find_latest_colmap_final(pipeline_output)
    )
    output = (
        args.output.expanduser().resolve()
        if args.output else pipeline_output / "comparison" / "alignment_report.json"
    )
    frame_path = pipeline_output / "resplat" / "reconstruction_frame.json"
    for path in (
        scene, frame_path, colmap_final / "cameras.txt", colmap_final / "images.txt"
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    with frame_path.open(encoding="utf-8") as f:
        frame_info = json.load(f)
    reference_c2w = np.asarray(
        frame_info["reference_c2w_original"], dtype=np.float64
    )
    world_to_model = np.linalg.inv(reference_c2w)

    scene_data = load_colmap_scene(
        str(scene), sparse_dir=args.sparse_dir, images_dir=args.images_dir
    )
    original_by_name = {
        Path(name).name: (scene_data["c2w"][index].astype(np.float64),
                          scene_data["intrinsics"][index].astype(np.float64))
        for index, name in enumerate(scene_data["image_names"])
    }
    final_images = read_extrinsics_text(colmap_final / "images.txt")
    final_cameras = read_intrinsics_text(colmap_final / "cameras.txt")
    final_by_name = {
        Path(image.name).name: image for image in final_images.values()
    }
    names = sorted(set(original_by_name) & set(final_by_name))
    if len(names) < 3:
        raise ValueError(
            f"Need at least 3 matching image names, found {len(names)}"
        )

    expected_poses = []
    final_poses = []
    for name in names:
        original_c2w, _ = original_by_name[name]
        expected_poses.append(world_to_model @ original_c2w)
        final_poses.append(image_c2w(final_by_name[name]))
    expected_poses = np.stack(expected_poses)
    final_poses = np.stack(final_poses)

    expected_centers = expected_poses[:, :3, 3]
    final_centers = final_poses[:, :3, 3]
    direct_center_errors = np.linalg.norm(
        expected_centers - final_centers, axis=1
    )
    direct_rotation_errors = [
        rotation_error_degrees(expected[:3, :3], final[:3, :3])
        for expected, final in zip(expected_poses, final_poses)
    ]

    scale, rotation, translation = fit_similarity(
        final_centers, expected_centers
    )
    aligned_centers = (
        scale * (rotation @ final_centers.T).T + translation
    )
    aligned_center_errors = np.linalg.norm(
        expected_centers - aligned_centers, axis=1
    )
    aligned_rotation_errors = [
        rotation_error_degrees(
            expected[:3, :3], rotation @ final[:3, :3]
        )
        for expected, final in zip(expected_poses, final_poses)
    ]
    scene_radius = float(np.sqrt(np.mean(np.sum(
        (expected_centers - expected_centers.mean(axis=0)) ** 2, axis=1
    ))))
    normalized_center_rmse = (
        float(np.sqrt(np.mean(aligned_center_errors ** 2)) / scene_radius)
        if scene_radius > 0 else None
    )

    intrinsic_relative_errors = []
    unsupported_camera_models = set()
    for name in names:
        _, original_k = original_by_name[name]
        final_image = final_by_name[name]
        final_camera = final_cameras[final_image.camera_id]
        final_k = normalized_intrinsics(final_camera)
        if final_k is None:
            unsupported_camera_models.add(final_camera.model)
            continue
        values = (
            np.array([final_k[0, 0], final_k[1, 1], final_k[0, 2], final_k[1, 2]])
            - np.array([original_k[0, 0], original_k[1, 1],
                        original_k[0, 2], original_k[1, 2]])
        )
        denominator = np.maximum(np.abs([
            original_k[0, 0], original_k[1, 1],
            original_k[0, 2], original_k[1, 2],
        ]), 1e-12)
        intrinsic_relative_errors.append(np.abs(values) / denominator)
    intrinsic_relative_errors = (
        np.asarray(intrinsic_relative_errors)
        if intrinsic_relative_errors else np.empty((0, 4))
    )

    global_only = (
        normalized_center_rmse is not None
        and normalized_center_rmse < 1e-4
        and stats(aligned_rotation_errors)["max"] < 0.01
    )
    diagnosis = (
        "global_similarity_only"
        if global_only
        else "per_view_pose_changes_remain_after_global_alignment"
    )
    report = {
        "schema_version": 1,
        "scene": str(scene),
        "colmap_final": str(colmap_final),
        "matched_image_count": len(names),
        "matched_image_names": names,
        "mapping": "resplat_model_point = scale * rotation @ reconstruction_point + translation",
        "reconstruction_to_resplat_similarity": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "scene_radius": scene_radius,
        "direct_center_error": stats(direct_center_errors),
        "direct_rotation_error_degrees": stats(direct_rotation_errors),
        "after_similarity_center_error": stats(aligned_center_errors),
        "after_similarity_center_rmse_over_scene_radius": normalized_center_rmse,
        "after_similarity_rotation_error_degrees": stats(aligned_rotation_errors),
        "intrinsics": {
            "supported_comparison_count": len(intrinsic_relative_errors),
            "unsupported_models": sorted(unsupported_camera_models),
            "mean_absolute_relative_error_fx_fy_cx_cy": (
                intrinsic_relative_errors.mean(axis=0).tolist()
                if len(intrinsic_relative_errors) else None
            ),
            "max_absolute_relative_error_fx_fy_cx_cy": (
                intrinsic_relative_errors.max(axis=0).tolist()
                if len(intrinsic_relative_errors) else None
            ),
        },
        "diagnosis": diagnosis,
        "global_similarity_is_sufficient": global_only,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nReconstruction/ReSplat alignment diagnosis")
    print(f"  COLMAP final: {colmap_final}")
    print(f"  Matched cameras: {len(names)}")
    print(f"  Sim(3) scale: {scale:.9f}")
    print(
        "  Direct center RMSE: "
        f"{report['direct_center_error']['rmse']:.6g}"
    )
    print(
        "  Aligned center RMSE / scene radius: "
        f"{normalized_center_rmse:.6g}"
    )
    print(
        "  Aligned rotation error mean/max: "
        f"{report['after_similarity_rotation_error_degrees']['mean']:.6g} / "
        f"{report['after_similarity_rotation_error_degrees']['max']:.6g} deg"
    )
    print(f"  Diagnosis: {diagnosis}")
    print(f"  Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
