#!/usr/bin/env python3
"""One-command black-box ReSplat inference for <scene>/image/*.

Run from anywhere:
    python scripts/run_scene.py /path/to/scene --gpu 0

Only wraps COLMAP and the upstream scripts/infer_colmap.py; does not alter ReSplat.
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
    parser.add_argument("scene_dir", type=Path, help="Scene directory containing image/")
    parser.add_argument("--gpu", default="0", help="CUDA device index, e.g. 0")
    parser.add_argument("--colmap", default="colmap", help="COLMAP binary")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--num_refine", type=int, default=4, help="Official recurrent iteration count")
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
    if shutil.which(args.colmap) is None:
        parser.error(f"COLMAP executable not found: {args.colmap}")

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
    mapping = prepare_images(inputs, prepared)
    (work / "image_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    timing["preprocess_s"] = round(time.perf_counter() - start, 3)

    colmap_start = time.perf_counter()
    if args.force_colmap:
        if sfm.exists():
            shutil.rmtree(sfm)
    if not has_sparse(undistorted / "sparse/0"):
        sfm.mkdir(parents=True, exist_ok=True)
        sparse_root.mkdir(parents=True, exist_ok=True)
        if not has_sparse(sparse_root / "0"):
            # Recreate the feature database when reconstruction is incomplete.
            if db.exists():
                db.unlink()
            run([args.colmap, "feature_extractor",
                 "--database_path", db, "--image_path", prepared,
                 "--ImageReader.camera_model", "SIMPLE_RADIAL",
                 "--ImageReader.single_camera", "1",
                 "--SiftExtraction.use_gpu", "1"], log)
            run([args.colmap, f"{args.matcher}_matcher",
                 "--database_path", db, "--SiftMatching.use_gpu", "1"], log)
            run([args.colmap, "mapper",
                 "--database_path", db, "--image_path", prepared,
                 "--output_path", sparse_root], log)
        models = sorted(p for p in sparse_root.iterdir() if p.is_dir() and has_sparse(p))
        if not models:
            raise RuntimeError("COLMAP failed to register images. Check overlap, blur and COLMAP log.")
        # Select the largest reconstruction (by registered images, via model_analyzer if available).
        model = models[0]
        if len(models) > 1:
            model = max(models, key=lambda p: (p / "images.bin").stat().st_size
                        if (p / "images.bin").exists() else 0)
        run([args.colmap, "image_undistorter",
             "--image_path", prepared, "--input_path", model,
             "--output_path", undistorted, "--output_type", "COLMAP"], log)
        # COLMAP image_undistorter writes sparse/ in some versions; normalize to sparse/0.
        if has_sparse(undistorted / "sparse") and not has_sparse(undistorted / "sparse/0"):
            (undistorted / "sparse/0").mkdir(parents=True, exist_ok=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin",
                         "cameras.txt", "images.txt", "points3D.txt"):
                src = undistorted / "sparse" / name
                if src.exists():
                    shutil.move(str(src), str(undistorted / "sparse/0" / name))
    if not has_sparse(undistorted / "sparse/0"):
        raise RuntimeError(f"Undistorted COLMAP model not found: {undistorted / 'sparse/0'}")
    timing["colmap_s"] = round(time.perf_counter() - colmap_start, 3)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        sys.executable, "scripts/infer_colmap.py",
        "--scene_path", undistorted,
        "--images_dir", "images",
        "--sparse_dir", "sparse/0",
        "--model_preset", "dl3dv_8v_512x960",
        "--checkpoint", CHECKPOINT,
        "--num_context", "8",
        "--num_refine", str(args.num_refine),
        "--context_selection", args.selection,
        "--start_frame", "0",
        "--frame_distance", str(len(mapping)),
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
        "num_input_images": len(mapping), "num_context": 8,
        "num_refine": args.num_refine, "selection": args.selection,
        "matcher": args.matcher, "gpu": args.gpu,
        "timing": timing,
    }, indent=2), encoding="utf-8")
    print(f"\n[success] All results: {results}")


if __name__ == "__main__":
    main()
