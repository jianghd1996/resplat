#!/usr/bin/env python3
"""Run the DL3DV competition baseline on one COLMAP scene.

The ReSplat forward path remains untouched. The inference adapter performs
native-resolution final rendering and sends the exported PLY to Module B.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.competition.module_b import run_module_b
from src.competition.scoring import build_report


DEFAULT_PRESET = "dl3dv_8v_512x960"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command ReSplat + Module B competition pipeline."
    )
    parser.add_argument("--scene", required=True, type=Path,
                        help="COLMAP scene folder containing images_4/ and sparse/0/.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for PLY, rendered PNGs, and report.")
    parser.add_argument("--model-preset", default=DEFAULT_PRESET,
                        help=f"Existing infer_colmap.py preset (default: {DEFAULT_PRESET}).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional checkpoint override; otherwise preset default is used.")
    parser.add_argument("--num-context", type=int, default=None,
                        help="Optional context-view override. Keep fixed across scenes.")
    parser.add_argument("--num-refine", type=int, default=None,
                        help="Optional ReSplat preset refinement override.")
    parser.add_argument("--max-resolution", type=int, default=None,
                        help="Optional maximum model input resolution override (long side).")
    parser.add_argument("--context-selection", choices=("fps", "uniform", "random"),
                        default="fps",
                        help="Context view selection strategy.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducible random context selection.")
    parser.add_argument("--images-dir", default="images_4",
                        help="Image directory relative to --scene.")
    parser.add_argument("--sparse-dir", default="sparse/0",
                        help="COLMAP model directory relative to --scene.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="First frame in the fixed inference/evaluation window.",
    )
    parser.add_argument(
        "--frame-distance", type=int, default=60,
        help="Number of consecutive frames in the inference/evaluation window.",
    )
    parser.add_argument(
        "--evaluation-count", type=int, default=None,
        help="Fixed stratified held-out evaluation view count.",
    )
    parser.add_argument(
        "--evaluation-seed", type=int, default=2026,
        help="Seed for fixed held-out evaluation view selection.",
    )
    parser.add_argument("--t-iter-sec", type=float, default=None,
                        help="Optional classic 3DGS seconds/iteration for the separate official Module B hard-limit check.")
    parser.add_argument(
        "--module-b", choices=("none", "identity", "reconstruction_full"),
        default="identity",
        help=(
            "Module B implementation. Use none to export ReSplat and optional "
            "Reconstruction_full inputs without touching module_b/."
        ),
    )
    parser.add_argument(
        "--export-reconstruction-init", action="store_true",
        help=("Export gaussians_model_frame.ply and reconstruction_frame.json "
              "for running Reconstruction_full later in another environment."),
    )
    parser.add_argument("--reconstruction-code-root", type=Path, default=None,
                        help="Parent directory containing Reconstruction_full; defaults to the repo root.")
    parser.add_argument("--reconstruction-mode", default="scene",
                        help="mode passed to Reconstruction_full.")
    parser.add_argument("--reconstruction-circle", type=int, default=0,
                        help="circle passed to Reconstruction_full (default: 0).")
    parser.add_argument("--reconstruction-max-frames", type=int, default=60,
                        help="max_frames passed to Reconstruction_full (default: 60).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the ReSplat command without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parent

    if not scene.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene}")
    required_paths = [scene / args.images_dir, scene / args.sparse_dir]
    missing = [str(path) for path in required_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Scene is missing required COLMAP folders: " + ", ".join(missing)
        )
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    if args.frame_distance <= 0:
        raise ValueError("--frame-distance must be positive.")
    if args.evaluation_count is not None and args.evaluation_count <= 0:
        raise ValueError("--evaluation-count must be positive.")
    if args.t_iter_sec is not None and args.t_iter_sec <= 0:
        raise ValueError("--t-iter-sec must be positive.")
    if args.module_b == "reconstruction_full":
        code_root = (args.reconstruction_code_root or repo_root).expanduser().resolve()
        if not ((code_root / "Reconstruction_full").is_dir()
                or (code_root / "Reconstruction_full.py").is_file()):
            raise FileNotFoundError(
                "Reconstruction_full is not available under "
                f"{code_root}. Pass --reconstruction-code-root to its parent folder."
            )

    output.mkdir(parents=True, exist_ok=True)
    inference_dir = output / "resplat"
    cmd = [
        sys.executable, "-u", str(repo_root / "scripts" / "infer_colmap.py"),
        "--scene_path", str(scene),
        "--output_dir", str(inference_dir),
        "--model_preset", args.model_preset,
        "--device", args.device,
        "--images_dir", args.images_dir,
        "--sparse_dir", args.sparse_dir,
        "--save_ply",
        "--save_images",
        "--max_save_images", "0",
        "--context_selection", args.context_selection,
        "--seed", str(args.seed),
        "--start_frame", str(args.start_frame),
        "--frame_distance", str(args.frame_distance),
    ]
    if args.evaluation_count is not None:
        cmd.extend([
            "--evaluation_count", str(args.evaluation_count),
            "--evaluation_seed", str(args.evaluation_seed),
        ])
    if args.module_b == "reconstruction_full" or args.export_reconstruction_init:
        cmd.append("--save_reconstruction_init")
    if args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint.expanduser().resolve())])
    if args.num_context is not None:
        cmd.extend(["--num_context", str(args.num_context)])
    if args.num_refine is not None:
        cmd.extend(["--num_refine", str(args.num_refine)])
    if args.max_resolution is not None:
        cmd.extend(["--max_resolution", str(args.max_resolution)])

    print("ReSplat inference command:")
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if args.dry_run:
        return 0

    # The model forward path is unchanged. The COLMAP adapter performs
    # Module A's FPS selection, inference, native-resolution rendering,
    # output saving, and metric evaluation in one subprocess.
    pipeline_start = time.perf_counter()
    subprocess.run(cmd, cwd=repo_root, check=True)
    inference_process_wall_seconds = time.perf_counter() - pipeline_start

    stage_timing_path = inference_dir / "scene_timing.json"
    if not stage_timing_path.is_file():
        raise FileNotFoundError(
            f"Missing warm-scene timing file {stage_timing_path}; "
            "update scripts/infer_colmap.py from the competition branch."
        )
    with stage_timing_path.open(encoding="utf-8") as f:
        stage_timing = json.load(f)
    model_load_seconds = float(stage_timing["model_load_seconds"])
    scene_processing_seconds = float(stage_timing["scene_processing_seconds"])
    quality_metric_seconds = float(
        stage_timing.get("quality_metric_evaluation_seconds", 0.0)
    )
    output_generation_seconds = max(
        0.0, scene_processing_seconds - quality_metric_seconds
    )

    source_ply = inference_dir / "gaussians.ply"
    if not source_ply.is_file():
        raise FileNotFoundError(
            f"ReSplat did not produce {source_ply}; inspect the inference log."
        )

    module_b_start = time.perf_counter()
    module_b_dir = output / "module_b"
    if args.module_b == "none":
        module_b_ply = source_ply
        module_b_metadata = {
            "name": "none",
            "is_quality_optimization": False,
            "input_ply": str(source_ply),
            "output_ply": None,
        }
    elif args.module_b == "identity":
        module_b_dir.mkdir(parents=True, exist_ok=True)
        module_b_ply, module_b_metadata = run_module_b(
            source_ply=source_ply,
            output_ply=module_b_dir / "gaussians.ply",
            method="identity",
        )
    else:
        module_b_dir.mkdir(parents=True, exist_ok=True)
        from src.competition.reconstruction_full_adapter import run_reconstruction_full

        init_ply = inference_dir / "gaussians_model_frame.ply"
        frame_path = inference_dir / "reconstruction_frame.json"
        if not init_ply.is_file() or not frame_path.is_file():
            raise FileNotFoundError(
                "Missing model-frame initialization PLY or reference camera transform "
                "for Reconstruction_full."
            )
        with frame_path.open(encoding="utf-8") as f:
            frame_info = json.load(f)
        reconstruction_metrics_path = inference_dir / "metrics.json"
        with reconstruction_metrics_path.open(encoding="utf-8") as f:
            reconstruction_metrics = json.load(f)
        selected_image_names = reconstruction_metrics.get("config", {}).get(
            "evaluation_image_names", []
        )
        if not selected_image_names:
            raise ValueError(
                "ReSplat metrics do not contain evaluation_image_names."
            )
        module_b_ply, module_b_metadata = run_reconstruction_full(
            initial_ply=init_ply,
            scene_path=scene,
            images_dir=args.images_dir,
            sparse_dir=args.sparse_dir,
            work_dir=module_b_dir / "reconstruction_work",
            export_dir=module_b_dir / "reconstruction_export",
            output_ply=module_b_dir / "gaussians.ply",
            code_root=args.reconstruction_code_root or repo_root,
            reference_c2w_original=frame_info["reference_c2w_original"],
            mode=args.reconstruction_mode,
            circle=args.reconstruction_circle,
            max_frames=args.reconstruction_max_frames,
            selected_image_names=selected_image_names,
        )
    if args.module_b == "none":
        refined_ply = source_ply
    else:
        refined_ply = output / "gaussians_refined.ply"
        shutil.copy2(module_b_ply, refined_ply)
    module_b_seconds = time.perf_counter() - module_b_start
    end_to_end_seconds = output_generation_seconds + module_b_seconds
    evaluation_end_to_end_seconds = scene_processing_seconds + module_b_seconds

    metrics_path = inference_dir / "metrics.json"
    metrics = None
    metrics_config = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            metrics_data = json.load(f)
        metrics = metrics_data.get("mean")
        metrics_config = metrics_data.get("config", {})

    report = build_report(
        scene=str(scene),
        output=str(output),
        preset=args.model_preset,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        module_b=module_b_metadata,
        inference_process_wall_seconds=inference_process_wall_seconds,
        model_load_seconds=model_load_seconds,
        scene_processing_seconds=scene_processing_seconds,
        quality_metric_seconds=quality_metric_seconds,
        output_generation_seconds=output_generation_seconds,
        module_b_seconds=module_b_seconds,
        end_to_end_seconds=end_to_end_seconds,
        evaluation_end_to_end_seconds=evaluation_end_to_end_seconds,
        t_iter_sec=args.t_iter_sec,
        metrics=metrics,
        rendered_png_dir=str(inference_dir / "rendered"),
        ply_path=str(refined_ply),
    )
    report["model"]["forward_input_resolution"] = stage_timing.get(
        "model_input_resolution"
    )
    report["quality"]["input_resolution"] = stage_timing.get("input_resolution")
    report["quality"]["rendered_resolution"] = stage_timing.get("output_resolution")
    report["quality"]["metric_resolution"] = stage_timing.get("input_resolution")
    report["quality"]["metrics_compare_against_original_input_images"] = True
    report["quality"]["metrics_source_ply"] = str(source_ply)
    report["quality"]["metrics_are_post_module_b"] = args.module_b in ("none", "identity")
    if args.module_b == "reconstruction_full":
        report["score"]["is_provisional_pre_module_b_quality"] = True
        report["score"]["note"] = (
            "Metrics in this report describe the original ReSplat PLY. "
            "The Reconstruction_full output PLY is saved separately and has not "
            "yet been re-rendered for post-optimization metrics."
        )
    report["quality"]["context_selection"] = metrics_config.get(
        "context_selection", args.context_selection
    )
    report["quality"]["selection_seed"] = metrics_config.get("selection_seed", args.seed)
    report["quality"]["target_selection"] = metrics_config.get("target_selection", "all")
    report["quality"]["evaluation_view_count"] = metrics_config.get("num_target")
    report["quality"]["evaluation_includes_context_views"] = metrics_config.get(
        "evaluation_includes_context_views", True
    )
    report["quality"]["evaluation_image_names"] = metrics_config.get(
        "evaluation_image_names", []
    )
    report["quality"]["context_image_names"] = metrics_config.get(
        "context_image_names", []
    )

    report_path = output / "performance_quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline complete: {output}")
    print(f"  PLY: {refined_ply}")
    if args.export_reconstruction_init:
        print(f"  Reconstruction_full init PLY: {inference_dir / 'gaussians_model_frame.ply'}")
        print(f"  Reconstruction frame: {inference_dir / 'reconstruction_frame.json'}")
    if args.module_b == "reconstruction_full":
        print("  Metrics in this report are the ReSplat pre-optimization baseline.")
    print(f"  PNG renders: {inference_dir / 'rendered'}")
    print(f"  Render/metric resolution: {stage_timing.get('output_resolution')}")
    print(f"  Report: {report_path}")
    print(f"  Model load (excluded from warm target): {model_load_seconds:.3f}s")
    print(f"  Scene processing incl. metric evaluation: {scene_processing_seconds:.3f}s")
    print(f"  Quality metric evaluation (reported separately): {quality_metric_seconds:.3f}s")
    print(f"  Data-to-saved-output (A + ReSplat, metric eval excluded): {output_generation_seconds:.3f}s")
    print(f"  End-to-end warm (A + ReSplat + B): {end_to_end_seconds:.3f}s")
    print(f"  End-to-end including metric evaluation: {evaluation_end_to_end_seconds:.3f}s")
    print(f"  Cold subprocess wall time: {inference_process_wall_seconds:.3f}s")
    print(f"  30s practical target: {report['practical_runtime_target']['status']}")
    if report["score"]["total"] is None:
        print("  Competition score: quality metrics are unavailable")
    else:
        print(f"  Competition score (30s total-time basis): {report['score']['total']:.2f}/100")
    print(f"  30s total-time score: {report['time_budget']['score_status']}")
    print(f"  Official Module B hard-limit check: "
          f"{report['time_budget']['official_module_b_constraint']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

