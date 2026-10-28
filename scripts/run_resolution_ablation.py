#!/usr/bin/env python3
"""Compare low and high model input resolutions with one fixed ReSplat checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PRESET = "dl3dv_8v_512x960"
NUM_CONTEXT = 8
STRATEGY = "fps"
SUMMARY_FIELDS = [
    "status",
    "model_preset",
    "num_context",
    "context_selection",
    "max_resolution",
    "model_input_resolution",
    "rendered_resolution",
    "metric_resolution",
    "psnr",
    "ssim",
    "lpips",
    "warm_end_to_end_seconds",
    "total_score",
    "evaluation_view_count",
    "same_view_sets",
    "context_image_names",
    "report_path",
    "log_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolution ablation for the 8-view high-resolution DL3DV ReSplat checkpoint."
    )
    parser.add_argument("--scene", required=True, type=Path,
                        help="COLMAP scene directory; the default first-60-frame subset is evaluated.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Root directory for per-resolution outputs and summaries.")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[448, 960],
                        help="Maximum long-side model input resolutions to compare (default: 448 960).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images-dir", default="images_4")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--num-refine", type=int, default=None)
    parser.add_argument("--t-iter-sec", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_report(report_path: Path) -> dict:
    with report_path.open(encoding="utf-8") as f:
        return json.load(f)


def make_row(report_path: Path, log_path: Path, max_resolution: int,
             status: str, error: str = "") -> dict:
    row = {
        "status": status,
        "model_preset": PRESET,
        "num_context": NUM_CONTEXT,
        "context_selection": STRATEGY,
        "max_resolution": max_resolution,
        "model_input_resolution": "",
        "rendered_resolution": "",
        "metric_resolution": "",
        "psnr": "",
        "ssim": "",
        "lpips": "",
        "warm_end_to_end_seconds": "",
        "total_score": "",
        "evaluation_view_count": "",
        "same_view_sets": "",
        "context_image_names": "",
        "report_path": str(report_path),
        "log_path": str(log_path),
        "error": error,
    }
    if report_path.is_file():
        report = read_report(report_path)
        quality = report.get("quality", {})
        metrics = quality.get("metrics") or {}
        row.update({
            "model_input_resolution": report.get("model", {}).get(
                "forward_input_resolution", ""
            ),
            "rendered_resolution": quality.get("rendered_resolution", ""),
            "metric_resolution": quality.get("metric_resolution", ""),
            "psnr": metrics.get("psnr", ""),
            "ssim": metrics.get("ssim", ""),
            "lpips": metrics.get("lpips", ""),
            "warm_end_to_end_seconds": report.get("timing", {}).get(
                "warm_end_to_end_seconds", ""
            ),
            "total_score": report.get("score", {}).get("total", ""),
            "evaluation_view_count": quality.get("evaluation_view_count", ""),
            "context_image_names": json.dumps(
                quality.get("context_image_names", []), ensure_ascii=False
            ),
        })
    return row


def write_summaries(output: Path, rows: list[dict]) -> None:
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
    if not args.resolutions or any(value <= 0 for value in args.resolutions):
        raise ValueError("--resolutions must contain positive values.")
    if len(set(args.resolutions)) != len(args.resolutions):
        raise ValueError("--resolutions must not contain duplicates.")

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    reference_context_names = None
    reference_evaluation_names = None
    failed = 0

    print(f"Resolution ablation: {len(args.resolutions)} run(s)")
    print(f"Fixed checkpoint: {PRESET}; context: {NUM_CONTEXT} views; strategy: {STRATEGY}")
    print("The same scene subset is rendered and evaluated at native image resolution.")

    for index, max_resolution in enumerate(args.resolutions, start=1):
        run_name = f"{PRESET}_fps_ctx8_maxres{max_resolution}"
        run_dir = output / run_name
        report_path = run_dir / "performance_quality_report.json"
        log_path = run_dir / "run.log"
        command = [
            sys.executable, "-u", str(pipeline_script),
            "--scene", str(scene),
            "--output", str(run_dir),
            "--model-preset", PRESET,
            "--num-context", str(NUM_CONTEXT),
            "--context-selection", STRATEGY,
            "--max-resolution", str(max_resolution),
            "--device", args.device,
            "--images-dir", args.images_dir,
            "--sparse-dir", args.sparse_dir,
        ]
        if args.num_refine is not None:
            command.extend(["--num-refine", str(args.num_refine)])
        if args.t_iter_sec is not None:
            command.extend(["--t-iter-sec", str(args.t_iter_sec)])

        print(f"\n[{index}/{len(args.resolutions)}] max_resolution={max_resolution}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue

        if args.skip_existing and report_path.is_file():
            print(f"  Reusing {report_path}")
            row = make_row(report_path, log_path, max_resolution, "reused")
        else:
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
                row = make_row(report_path, log_path, max_resolution, "complete")
            else:
                failed += 1
                row = make_row(
                    report_path, log_path, max_resolution, "failed",
                    error=f"pipeline exit={return_code}",
                )
                print(f"  FAILED ({row['error']}); see {log_path}")

        if report_path.is_file():
            report = read_report(report_path)
            quality = report.get("quality", {})
            context_names = quality.get("context_image_names")
            evaluation_names = quality.get("evaluation_image_names")
            if reference_context_names is None:
                reference_context_names = context_names
                reference_evaluation_names = evaluation_names
                row["same_view_sets"] = bool(context_names and evaluation_names)
            else:
                same = (
                    context_names == reference_context_names
                    and evaluation_names == reference_evaluation_names
                )
                row["same_view_sets"] = same
                if not same:
                    row["status"] = "failed"
                    row["error"] = "input/evaluation view lists differ across resolutions"
                    failed += 1
            print(
                f"  input={row['model_input_resolution']} "
                f"PSNR={row['psnr']} SSIM={row['ssim']} LPIPS={row['lpips']} "
                f"time={row['warm_end_to_end_seconds']}s "
                f"same_views={row['same_view_sets']}"
            )
        rows.append(row)
        write_summaries(output, rows)

    if args.dry_run:
        return 0
    print(f"\nResolution ablation finished: {len(rows) - failed}/{len(rows)} successful.")
    print(f"CSV summary: {output / 'summary.csv'}")
    print(f"JSON summary: {output / 'summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
