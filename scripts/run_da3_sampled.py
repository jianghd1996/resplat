#!/usr/bin/env python3
"""Sample a scene's images before running the custom DA3 pipeline.

Default: python scripts/run_da3_sampled.py
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ===================== User configuration =====================
WORK_DIR = ROOT / "datasets/test"
MAX_VIEWS = 64
SAMPLE_MODE = "head"  # "head": first 64 consecutive frames; "uniform": global coverage
SAMPLE_START = 0
DA3_MODEL_PATH = ROOT / "Cloud_Models/Saliency-Face-DA3/besthumanmodel_lr1e_6_ckpt2.pt"
VIEW_NUM = 81
CROP_RATIO = 1.0
EXTRA_VIEW = [False, False]
# ==============================================================


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def sample_indices(n, max_views, mode="head", start=0):
    if mode == "head":
        if start < 0 or start >= n:
            raise ValueError(f"sample start {start} outside [0, {n})")
        return list(range(start, min(n, start + max_views)))
    if mode == "uniform":
        if n <= max_views:
            return list(range(n))
        return [(i * (n - 1)) // (max_views - 1) for i in range(max_views)]
    raise ValueError(f"Unknown sample mode: {mode}")


def prepare_sampled_images(image_dir, output_dir, max_views=64, force=False, mode=SAMPLE_MODE, start=SAMPLE_START):
    if max_views < 2:
        raise ValueError("max_views must be >= 2")
    files = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=natural_key,
    )
    if not files:
        raise FileNotFoundError(f"No supported images in {image_dir}")
    indices = sample_indices(len(files), max_views, mode=mode, start=start)
    selected = [files[i] for i in indices]
    manifest = {
        "source_dir": str(image_dir.resolve()),
        "sample_mode": mode,
        "sample_start": start,
        "source_count": len(files),
        "selected_count": len(selected),
        "selected_indices": indices,
        "files": [
            {
                "index": i,
                "original": files[i].name,
                "sampled": f"{j:06d}.jpg",
                "size": files[i].stat().st_size,
                "mtime_ns": files[i].stat().st_mtime_ns,
            }
            for j, i in enumerate(indices)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "da3_sample_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    expected = {f"{j:06d}.jpg" for j in range(len(selected))}
    cached = (
        not force
        and previous == manifest
        and {p.name for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"} == expected
    )
    if not cached:
        for old in output_dir.iterdir():
            if old.is_file() or old.is_symlink():
                old.unlink()
        for j, src in enumerate(selected):
            dst = output_dir / f"{j:06d}.jpg"
            with Image.open(src) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, quality=95, subsampling=0)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[DA3 sample] {len(files)} -> {len(selected)} images; cached={cached}", flush=True)
    print(f"[DA3 sample] indices: {indices}", flush=True)
    return selected, cached


@dataclass
class DA3Params:
    data_path: str
    colmap_path: str
    gs_render_vid_path: str
    view_num: int
    crop_ratio: float
    extra_view: list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", nargs="?", type=Path, default=WORK_DIR)
    parser.add_argument("--max-views", type=int, default=MAX_VIEWS)
    parser.add_argument("--sample-mode", choices=("head", "uniform"), default=SAMPLE_MODE)
    parser.add_argument("--sample-start", type=int, default=SAMPLE_START)
    parser.add_argument("--da3-model", type=Path, default=DA3_MODEL_PATH)
    parser.add_argument("--view-num", type=int, default=VIEW_NUM)
    parser.add_argument("--crop-ratio", type=float, default=CROP_RATIO)
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--force-da3", action="store_true")
    parser.add_argument("--sample-only", action="store_true", help="Only prepare images; do not load DA3")
    args = parser.parse_args()

    scene = args.scene_dir.expanduser().resolve()
    image_dir = scene / "image"
    if not image_dir.is_dir():
        parser.error(f"Missing image directory: {image_dir}")
    work = scene / "da3_work"
    sample_dir = work / "sampled_images"
    start = time.perf_counter()
    selected, cached = prepare_sampled_images(image_dir, sample_dir, args.max_views, args.force_sample, mode=args.sample_mode, start=args.sample_start)
    if args.sample_only:
        return

    colmap_path = scene / "sparse" / "0"
    render_path = scene / "3dgs_render" / "gs_render.mp4"
    colmap_path.mkdir(parents=True, exist_ok=True)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    has_cameras = (colmap_path / "cameras.bin").exists() or (colmap_path / "cameras.txt").exists()
    has_images = (colmap_path / "images.bin").exists() or (colmap_path / "images.txt").exists()
    # Only reuse the reconstruction if the sampled image manifest is unchanged.
    if not args.force_da3 and cached and has_cameras and has_images:
        print(f"[DA3] Existing reconstruction found: {colmap_path}; use --force-da3 to rerun", flush=True)
        return

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from DA3.run_da3_full_crop_dev import run_DA3

    model_path = args.da3_model.expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"DA3 checkpoint missing: {model_path}")
    da3 = run_DA3(str(model_path))
    result = da3.infer(
        args=DA3Params(
            data_path=str(sample_dir),
            colmap_path=str(colmap_path),
            gs_render_vid_path=str(render_path),
            view_num=args.view_num,
            crop_ratio=args.crop_ratio,
            extra_view=EXTRA_VIEW.copy(),
        )
    )
    print(f"[DA3] finished in {time.perf_counter() - start:.2f}s; result={result}", flush=True)
    print(f"[DA3] COLMAP: {colmap_path}", flush=True)


if __name__ == "__main__":
    main()
