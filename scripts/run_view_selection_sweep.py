#!/usr/bin/env python3
"""Run a reproducible ReSplat view-selection and context-count sweep.

By default this runs the four DL3DV checkpoints discussed for the competition
and the fps, uniform, and seeded-random strategies. Each checkpoint is tested
with its matching context count and the same scene frame range.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PRESET_CONTEXT_COUNTS = {
    "dl3dv_8v_512x960": 8,
    "dl3dv_16v_256x448": 16,
    "dl3dv_32v_256x448": 32,
    "dl3dv_8v_256x448_large": 8,
}
DEFAULT_PRESETS = list(PRESET_CONTEXT_COUNTS)
DEFAULT_STRATEGIES = ["fps", "uniform", "random"]
SUMMARY_FIELDS = [
    "status",
    "model_preset",
    "num_context",
    "context_selection",
    "seed",
    "psnr",
    "ssim",
    "lpips",
    "warm_end_to_end_seconds",
    "total_score",
    "evaluation_view_count",
    "report_path",
    "log_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ReSplat context selection strategies and input-view counts."
    )
    parser.add_argument("--scene", required=True, type=Path,
                        help="COLMAP scene directory; inference evaluates its fixed first 60 frames.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Root directory for per-run outputs and sweep summaries.")
    parser.add_argument("--model-presets", nargs="+", choices=DEFAULT_PRESETS,
                        default=DEFAULT_PRESETS,
                        help="Checkpoint presets to sweep; each preset uses its trained context count.")
    parser.add_argument("--strategies", nargs="+", choices=("fps", "uniform", "random"),
                        default=DEFAULT_STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Seeds for reproducible random selection; non-random strategies also get one run per seed.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--num-refine", type=int, default=None)
    parser.add_argument("--t-iter-sec", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse a run when its report already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the complete experiment matrix without launching inference.")
    return parser.parse_args()


def load_row(report_path: Path, preset: str, count: int, strategy: str,
             seed: int, log_path: Path, status: str = "complete",
             error: str = "") -> dict:
    row = {
        "status": status,
        "model_preset": preset,
        "num_context": count,
        "context_selection": strategy,
        "seed": seed,
        "psnr": "",
        "ssim": "",
        "lpips": "",
        "warm_end_to_end_seconds": "",
        "total_score": "",
        "evaluation_view_count": "",
        "report_path": str(report_path),
        "log_path": str(log_path),
        "error": error,
    }
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("quality", {}).get("metrics") or {}
        row.update({
            "psnr": metrics.get("psnr", ""),
            "ssim": metrics.get("ssim", ""),
            "lpips": metrics.get("lpips", ""),
            "warm_end_to_end_seconds": report.get("timing", {}).get(
                "warm_end_to_end_seconds", ""
            ),
            "total_score": report.get("score", {}).get("total", ""),
            "evaluation_view_count": report.get("quality", {}).get(
                "evaluation_view_count", ""
            ),
        })
    return row


def write_summaries(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    pipeline_script = repo_root / "run_pipeline.py"

    if not scene.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene}")
    for path in (scene / args.images_dir, scene / args.sparse_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Required scene directory does not exist: {path}")

    experiments = [
        (preset, PRESET_CONTEXT_COUNTS[preset], strategy, seed)
        for preset in args.model_presets
        for strategy in args.strategies
        for seed in args.seeds
    ]
    print(f"Experiment count: {len(experiments)}")
    print("Each run uses the same scene subset and scores every selected evaluation view.")
    rows: list[dict] = []

    for index, (preset, count, strategy, seed) in enumerate(experiments, start=1):
        run_name = f"{preset}_ctx{count}_{strategy}_seed{seed}"
        run_dir = output / run_name
        report_path = run_dir / "performance_quality_report.json"
        log_path = run_dir / "run.log"
        command = [
            sys.executable, "-u", str(pipeline_script),
            "--scene", str(scene),
            "--output", str(run_dir),
            "--model-preset", preset,
            "--num-context", str(count),
            "--context-selection", strategy,
            "--seed", str(seed),
            "--device", args.device,
            "--images-dir", args.images_dir,
            "--sparse-dir", args.sparse_dir,
        ]
        if args.num_refine is not None:
            command.extend(["--num-refine", str(args.num_refine)])
        if args.t_iter_sec is not None:
            command.extend(["--t-iter-sec", str(args.t_iter_sec)])

        print(f"\n[{index}/{len(experiments)}] {preset} | views={count} | "
              f"strategy={strategy} | seed={seed}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue

        if args.skip_existing and report_path.is_file():
            print(f"  Reusing existing report: {report_path}")
            rows.append(load_row(report_path, preset, count, strategy, seed,
                                 log_path, status="reused"))
            write_summaries(output, rows)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print("  " + line.rstrip(), flush=True)
                log_file.write(line)
            process.stdout.close()
            return_code = process.wait()
        if return_code == 0 and report_path.is_file():
            row = load_row(report_path, preset, count, strategy, seed, log_path)
            print(f"  PSNR={row['psnr']} SSIM={row['ssim']} "
                  f"LPIPS={row['lpips']} time={row['warm_end_to_end_seconds']}s")
        else:
            error = f"pipeline exit={return_code}"
            row = load_row(report_path, preset, count, strategy, seed,
                           log_path, status="failed", error=error)
            print(f"  FAILED ({error}); see {log_path}")
        rows.append(row)
        write_summaries(output, rows)

    if args.dry_run:
        return 0
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"\nSweep finished: {len(rows) - failed}/{len(rows)} completed, {failed} failed.")
    print(f"CSV summary: {output / 'summary.csv'}")
    print(f"JSON summary: {output / 'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
