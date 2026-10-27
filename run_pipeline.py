#!/usr/bin/env python3
"""Run the DL3DV competition baseline on one COLMAP scene.

ReSplat remains a black box: this wrapper calls scripts/infer_colmap.py unchanged,
then sends its PLY through the external Module B adapter.
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
    parser.add_argument("--images-dir", default="images_4",
                        help="Image directory relative to --scene.")
    parser.add_argument("--sparse-dir", default="sparse/0",
                        help="COLMAP model directory relative to --scene.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-iter-sec", type=float, default=None,
                        help="Measured classic 3DGS seconds/iteration on this machine.")
    parser.add_argument("--module-b", choices=("identity",), default="identity",
                        help="Module B implementation. identity is the reproducible baseline.")
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
    if args.t_iter_sec is not None and args.t_iter_sec <= 0:
        raise ValueError("--t-iter-sec must be positive.")

    output.mkdir(parents=True, exist_ok=True)
    inference_dir = output / "resplat"
    cmd = [
        sys.executable, str(repo_root / "scripts" / "infer_colmap.py"),
        "--scene_path", str(scene),
        "--output_dir", str(inference_dir),
        "--model_preset", args.model_preset,
        "--device", args.device,
        "--images_dir", args.images_dir,
        "--sparse_dir", args.sparse_dir,
        "--save_ply",
        "--save_images",
        "--max_save_images", "0",
    ]
    if args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint.expanduser().resolve())])
    if args.num_context is not None:
        cmd.extend(["--num_context", str(args.num_context)])
    if args.num_refine is not None:
        cmd.extend(["--num_refine", str(args.num_refine)])

    print("ReSplat inference command:")
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if args.dry_run:
        return 0

    # This is the unchanged upstream inference entry point. Its wall time is
    # recorded separately and is not counted against the Module B allowance.
    inference_start = time.perf_counter()
    subprocess.run(cmd, cwd=repo_root, check=True)
    inference_seconds = time.perf_counter() - inference_start

    source_ply = inference_dir / "gaussians.ply"
    if not source_ply.is_file():
        raise FileNotFoundError(
            f"ReSplat did not produce {source_ply}; inspect the inference log."
        )

    module_b_start = time.perf_counter()
    module_b_dir = output / "module_b"
    module_b_dir.mkdir(parents=True, exist_ok=True)
    module_b_ply, module_b_metadata = run_module_b(
        source_ply=source_ply,
        output_ply=module_b_dir / "gaussians.ply",
        method=args.module_b,
    )
    refined_ply = output / "gaussians_refined.ply"
    shutil.copy2(module_b_ply, refined_ply)
    module_b_seconds = time.perf_counter() - module_b_start

    metrics_path = inference_dir / "metrics.json"
    metrics = None
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f).get("mean")

    report = build_report(
        scene=str(scene),
        output=str(output),
        preset=args.model_preset,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        module_b=module_b_metadata,
        inference_seconds=inference_seconds,
        module_b_seconds=module_b_seconds,
        t_iter_sec=args.t_iter_sec,
        metrics=metrics,
        rendered_png_dir=str(inference_dir / "rendered"),
        ply_path=str(refined_ply),
    )
    report_path = output / "performance_quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline complete: {output}")
    print(f"  PLY: {refined_ply}")
    print(f"  PNG renders: {inference_dir / 'rendered'}")
    print(f"  Report: {report_path}")
    print(f"  Module B: {module_b_seconds:.6f}s")
    if report["score"]["total"] is None:
        print("  Competition score: partial (supply --t-iter-sec to score time efficiency)")
    else:
        print(f"  Competition score: {report['score']['total']:.2f}/100")
        print(f"  Module B budget status: {report['time_budget']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
