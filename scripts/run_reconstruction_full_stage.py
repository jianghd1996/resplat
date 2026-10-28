#!/usr/bin/env python3
"""Run Reconstruction_full as a separate, Docker-friendly pipeline stage."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.competition.reconstruction_full_adapter import run_reconstruction_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reconstruction_full from an exported ReSplat result."
    )
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument(
        "--pipeline-output", required=True, type=Path,
        help="Output folder previously passed to run_pipeline.py.",
    )
    parser.add_argument(
        "--reconstruction-code-root", type=Path, default=REPO_ROOT,
        help="Parent folder containing Reconstruction_full.",
    )
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--mode", default="scene")
    parser.add_argument("--circle", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output-ply", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    pipeline_output = args.pipeline_output.expanduser().resolve()
    inference_dir = pipeline_output / "resplat"
    module_b_dir = pipeline_output / "module_b"
    initial_ply = inference_dir / "gaussians_model_frame.ply"
    frame_path = inference_dir / "reconstruction_frame.json"
    metrics_path = inference_dir / "metrics.json"
    output_ply = (
        args.output_ply.expanduser().resolve()
        if args.output_ply else module_b_dir / "gaussians.ply"
    )

    for path in (scene, initial_ply, frame_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)
    with frame_path.open(encoding="utf-8") as f:
        frame_info = json.load(f)
    with metrics_path.open(encoding="utf-8") as f:
        metrics_info = json.load(f)
    selected_image_names = metrics_info.get("config", {}).get(
        "evaluation_image_names", []
    )
    if not selected_image_names:
        raise ValueError(
            f"{metrics_path} has no config.evaluation_image_names"
        )
    if len(selected_image_names) != args.max_frames:
        raise ValueError(
            f"Fixed evaluation set has {len(selected_image_names)} views, but "
            f"--max-frames is {args.max_frames}."
        )

    stage_start = time.perf_counter()
    result_ply, metadata = run_reconstruction_full(
        initial_ply=initial_ply,
        scene_path=scene,
        images_dir=args.images_dir,
        sparse_dir=args.sparse_dir,
        work_dir=module_b_dir / "reconstruction_work",
        export_dir=module_b_dir / "reconstruction_export",
        output_ply=output_ply,
        code_root=args.reconstruction_code_root,
        reference_c2w_original=frame_info["reference_c2w_original"],
        mode=args.mode,
        circle=args.circle,
        max_frames=args.max_frames,
        selected_image_names=selected_image_names,
    )
    metadata["stage_wall_seconds"] = round(time.perf_counter() - stage_start, 6)
    metadata["coordinate_frame"] = frame_info.get("coordinate_frame")
    metadata["reference_image_name"] = frame_info.get("reference_image_name")
    metadata_path = module_b_dir / "reconstruction_full_report.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nReconstruction_full stage complete")
    print(f"  Optimized PLY: {result_ply}")
    print(f"  Stage report: {metadata_path}")
    print(f"  Wall time: {metadata['stage_wall_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
