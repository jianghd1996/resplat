#!/usr/bin/env python3
"""Run custom DA3 sampling, pose estimation, and ReSplat inference end-to-end.

Run from anywhere:
    python scripts/run_scene.py /path/to/scene --gpu 0

Runs scripts/run_da3_sampled.py automatically; no COLMAP executable.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def run(cmd, log_path, cwd=None, env=None):
    print("\n[command]", " ".join(map(str, cmd)), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}. See {log_path}")


def prepare_images(input_dir, images_dir):
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    if len(files) < 8:
        raise ValueError(f"8-view model needs at least 8 images; found {len(files)} in {input_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for idx, file in enumerate(files):
        dest = images_dir / f"{idx:06d}.jpg"
        with Image.open(file) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if not dest.exists() or dest.stat().st_mtime < file.stat().st_mtime:
                im.save(dest, "JPEG", quality=95, subsampling=0)
        mapping.append({"prepared": dest.name, "original": file.name})
    # Remove stale converted images from earlier runs with larger input sets.
    keep = {x["prepared"] for x in mapping}
    for old in images_dir.glob("*.jpg"):
        if old.name not in keep:
            old.unlink()
    return mapping


def has_sparse(model_dir):
    return ((model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists()) or (
        (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", nargs="?", type=Path, default=REPO / "datasets/test", help="Scene directory containing image/")
    parser.add_argument("--max-views", type=int, default=64)
    parser.add_argument("--sample-mode", choices=("head", "uniform"), default="head")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--force-da3", action="store_true")
    parser.add_argument("--gpu", default=None, help="Physical GPU index; defaults to inherited CUDA_VISIBLE_DEVICES or GPU 0")
    parser.add_argument("--colmap", default="colmap", help="COLMAP binary")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--num_refine", type=int, default=4, help="Official recurrent iteration count")
    parser.add_argument("--num-context", type=int, default=64, help="Use all 64 sampled images as ReSplat context")
    parser.add_argument("--selection", choices=("fps", "uniform"), default="fps")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_colmap", action="store_true", help="Rebuild COLMAP reconstruction")
    parser.add_argument("--render_chunk_size", type=int, default=4)
    args = parser.parse_args()

    scene = args.scene_dir.expanduser().resolve()
    inputs = scene / "image"
    if not inputs.is_dir():
        parser.error(f"Missing input directory: {inputs}")
    if not CHECKPOINT.is_file():
        parser.error(f"Checkpoint not found: {CHECKPOINT}")
    # DA3 mode does not require the COLMAP executable.

    work = scene / "resplat_work"
    prepared = work / "prepared_images"
    sfm = work / "colmap"
    db = sfm / "database.db"
    sparse_root = sfm / "sparse"
    undistorted = sfm / "undistorted"
    results = scene / "resplat_results"
    log = work / "run.log"
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    timing = {}
    start = time.perf_counter()
    timing["preprocess_s"] = 0.0  # DA3 sampler handles image conversion.

    # Use the custom DA3 output, which already contains camera intrinsics/poses.
    # DA3 was run on sampled_images, so images and camera filenames must match.
    da3_sparse = scene / "sparse" / "0"
    da3_images = scene / "da3_work" / "sampled_images"
    da3_command = [sys.executable, "scripts/run_da3_sampled.py", str(scene),
                   "--max-views", str(args.max_views), "--sample-mode", args.sample_mode,
                   "--sample-start", str(args.sample_start)]
    # Always recompute DA3 poses: never reuse a previous reconstruction.
    da3_command.append("--force-da3")
    gpu = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in gpu:
        raise ValueError("Specify a single GPU via --gpu or CUDA_VISIBLE_DEVICES")
    gpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[GPU] DA3 and ReSplat will both use CUDA_VISIBLE_DEVICES={gpu}", flush=True)
    da3_start = time.perf_counter()
    run(da3_command, log, cwd=REPO, env=gpu_env)
    timing["da3_s"] = round(time.perf_counter() - da3_start, 3)
    if not has_sparse(da3_sparse):
        raise RuntimeError(f"DA3 COLMAP export missing: {da3_sparse}")
    if not da3_images.is_dir():
        raise RuntimeError(f"DA3 sampled images missing: {da3_images}")
    sampled = sorted(da3_images.glob("*.jpg"))
    if len(sampled) < args.num_context:
        raise RuntimeError(f"Requested {args.num_context} context views but only {len(sampled)} DA3 views exist")
    # ReSplat reads poses from sparse/0 and matching images from images/.
    # Symlinks avoid copying hundreds of MB; no COLMAP binary needed.
    undistorted = work / "da3_scene"
    (undistorted / "sparse").mkdir(parents=True, exist_ok=True)
    for name, target in (("images", da3_images), ("sparse/0", da3_sparse)):
        link = undistorted / name
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing directory: {link}")
        link.symlink_to(target.resolve(), target_is_directory=True)

    env = gpu_env
    command = [
        sys.executable, "scripts/infer_colmap.py",
        "--scene_path", undistorted,
        "--images_dir", "images",
        "--sparse_dir", "sparse/0",
        "--model_preset", "dl3dv_8v_512x960",
        "--checkpoint", CHECKPOINT,
        "--num_context", str(args.num_context),
        "--target_selection", "all",
        "--num_refine", str(args.num_refine),
        "--context_selection", args.selection,
        "--start_frame", "0",
        "--frame_distance", str(len(sampled)),
        "--render_chunk_size", str(args.render_chunk_size),
        "--output_dir", results,
        "--save_images", "--save_ply", "--save_depth",
        "--device", "cuda:0",
    ]
    if not args.skip_video:
        command.append("--save_video")
    if args.skip_eval:
        command.append("--no_eval")

    inference_start = time.perf_counter()
    try:
        run(command, log, cwd=REPO, env=env)
    finally:
        timing["inference_s"] = round(time.perf_counter() - inference_start, 3)
        timing["total_s"] = round(time.perf_counter() - start, 3)
        (results / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (results / "run_config.json").write_text(json.dumps({
        "scene": str(scene), "checkpoint": str(CHECKPOINT),
        "num_input_images": len(sampled), "num_context": args.num_context,
        "target_selection": "all",
        "sample_mode": args.sample_mode, "sample_start": args.sample_start,
        "num_refine": args.num_refine, "selection": args.selection,
        "pose_source": "DA3", "num_da3_views": len(sampled), "gpu": gpu,
        "timing": timing,
    }, indent=2), encoding="utf-8")
    print(f"\n[success] All results: {results}")


if __name__ == "__main__":
    main()
