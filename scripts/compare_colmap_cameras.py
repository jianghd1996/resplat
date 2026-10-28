#!/usr/bin/env python3
"""Check whether Reconstruction changed input COLMAP cameras or poses."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sparse", required=True, type=Path)
    parser.add_argument("--output-colmap", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--center-tolerance", type=float, default=1e-5)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1e-4)
    parser.add_argument("--intrinsics-relative-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def read_cameras_bin(path):
    cameras = {}
    with path.open("rb") as f:
        count, = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            model, num_params = CAMERA_MODELS[model_id]
            params = np.asarray(
                struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            )
            cameras[camera_id] = {
                "id": camera_id, "model": model, "width": width,
                "height": height, "params": params,
            }
    return cameras


def read_cameras_txt(path):
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = {
            "id": camera_id, "model": fields[1],
            "width": int(fields[2]), "height": int(fields[3]),
            "params": np.asarray(list(map(float, fields[4:]))),
        }
    return cameras


def read_cstring(f):
    value = bytearray()
    while True:
        byte = f.read(1)
        if byte == b"\0":
            return value.decode("utf-8")
        if not byte:
            raise EOFError("Truncated COLMAP image name")
        value.extend(byte)


def read_images_bin(path):
    images = {}
    with path.open("rb") as f:
        count, = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            values = struct.unpack("<i7di", f.read(64))
            image_id = values[0]
            name = read_cstring(f)
            point_count, = struct.unpack("<Q", f.read(8))
            f.seek(point_count * 24, 1)
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(values[1:5]),
                "tvec": np.asarray(values[5:8]),
                "camera_id": values[8],
                "name": name,
            }
    return images


def read_images_txt(path):
    images = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        image_id = int(fields[0])
        images[image_id] = {
            "id": image_id,
            "qvec": np.asarray(list(map(float, fields[1:5]))),
            "tvec": np.asarray(list(map(float, fields[5:8]))),
            "camera_id": int(fields[8]),
            "name": fields[9],
        }
        if index < len(lines):
            index += 1
    return images


def load_model(folder):
    folder = folder.resolve()
    if (folder / "cameras.bin").is_file() and (folder / "images.bin").is_file():
        return (
            read_cameras_bin(folder / "cameras.bin"),
            read_images_bin(folder / "images.bin"),
            "binary",
        )
    if (folder / "cameras.txt").is_file() and (folder / "images.txt").is_file():
        return (
            read_cameras_txt(folder / "cameras.txt"),
            read_images_txt(folder / "images.txt"),
            "text",
        )
    raise FileNotFoundError(
        f"Missing cameras/images COLMAP model under {folder}"
    )


def qvec_to_rotation(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose_values(image):
    w2c_rotation = qvec_to_rotation(image["qvec"])
    c2w_rotation = w2c_rotation.T
    center = -c2w_rotation @ image["tvec"]
    return c2w_rotation, center


def rotation_error_deg(first, second):
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalized_intrinsics(camera):
    model, params = camera["model"], camera["params"]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        fx, cx, cy = params[:3]
        fy = fx
    else:
        return None
    return np.asarray([
        fx / camera["width"], fy / camera["height"],
        cx / camera["width"], cy / camera["height"],
    ])


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "max": float(values.max()),
        "rmse": float(math.sqrt(np.mean(values ** 2))),
    }


def match_images(input_images, output_images):
    input_by_name = {Path(image["name"]).name: image for image in input_images.values()}
    output_by_name = {Path(image["name"]).name: image for image in output_images.values()}
    common_names = sorted(set(input_by_name) & set(output_by_name))
    if len(common_names) == len(input_images) == len(output_images):
        return [
            (input_by_name[name], output_by_name[name]) for name in common_names
        ], "exact_image_name"
    common_ids = sorted(set(input_images) & set(output_images))
    if len(common_ids) == len(input_images) == len(output_images):
        return [
            (input_images[image_id], output_images[image_id])
            for image_id in common_ids
        ], "image_id"
    return [], "unresolved"


def main():
    args = parse_args()
    input_cameras, input_images, input_format = load_model(args.input_sparse)
    output_cameras, output_images, output_format = load_model(args.output_colmap)
    pairs, mapping = match_images(input_images, output_images)

    center_errors, rotation_errors, intrinsic_errors, rows = [], [], [], []
    for input_image, output_image in pairs:
        input_rotation, input_center = pose_values(input_image)
        output_rotation, output_center = pose_values(output_image)
        center_error = float(np.linalg.norm(input_center - output_center))
        rotation_error = rotation_error_deg(input_rotation, output_rotation)
        center_errors.append(center_error)
        rotation_errors.append(rotation_error)

        input_k = normalized_intrinsics(input_cameras[input_image["camera_id"]])
        output_k = normalized_intrinsics(output_cameras[output_image["camera_id"]])
        intrinsic_error = None
        if input_k is not None and output_k is not None:
            relative = np.abs(output_k - input_k) / np.maximum(np.abs(input_k), 1e-12)
            intrinsic_error = float(relative.max())
            intrinsic_errors.append(intrinsic_error)
        rows.append({
            "input_name": input_image["name"],
            "output_name": output_image["name"],
            "input_image_id": input_image["id"],
            "output_image_id": output_image["id"],
            "center_error": center_error,
            "rotation_error_degrees": rotation_error,
            "max_intrinsics_relative_error": intrinsic_error,
        })

    fully_matched = len(pairs) == len(input_images) == len(output_images)
    unchanged = (
        fully_matched
        and max(center_errors, default=float("inf")) <= args.center_tolerance
        and max(rotation_errors, default=float("inf")) <= args.rotation_tolerance_deg
        and max(intrinsic_errors, default=0.0) <= args.intrinsics_relative_tolerance
    )
    if not fully_matched:
        status = "unverified_image_mapping_changed"
    elif unchanged:
        status = "pass_cameras_unchanged"
    else:
        status = "warning_cameras_changed"

    report = {
        "schema_version": 1,
        "input_sparse": str(args.input_sparse.resolve()),
        "output_colmap": str(args.output_colmap.resolve()),
        "input_format": input_format,
        "output_format": output_format,
        "input_camera_count": len(input_cameras),
        "output_camera_count": len(output_cameras),
        "input_image_count": len(input_images),
        "output_image_count": len(output_images),
        "image_mapping": mapping,
        "matched_image_count": len(pairs),
        "image_names_changed": any(
            Path(first["name"]).name != Path(second["name"]).name
            for first, second in pairs
        ),
        "center_error": summarize(center_errors),
        "rotation_error_degrees": summarize(rotation_errors),
        "intrinsics_max_relative_error": summarize(intrinsic_errors),
        "tolerances": {
            "center": args.center_tolerance,
            "rotation_degrees": args.rotation_tolerance_deg,
            "intrinsics_relative": args.intrinsics_relative_tolerance,
        },
        "status": status,
        "per_image": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nCOLMAP camera consistency check")
    print(f"  Input:  {args.input_sparse.resolve()} ({input_format})")
    print(f"  Output: {args.output_colmap.resolve()} ({output_format})")
    print(f"  Images: {len(input_images)} -> {len(output_images)}")
    print(f"  Mapping: {mapping}; matched {len(pairs)}")
    if fully_matched:
        print(f"  Center error max: {report['center_error']['max']:.9g}")
        print(
            "  Rotation error max: "
            f"{report['rotation_error_degrees']['max']:.9g} deg"
        )
        if report["intrinsics_max_relative_error"]:
            print(
                "  Intrinsics relative error max: "
                f"{report['intrinsics_max_relative_error']['max']:.9g}"
            )
    if status == "pass_cameras_unchanged":
        print("  PASS: Reconstruction preserved input cameras and poses.")
    elif status == "warning_cameras_changed":
        print("  WARNING: Reconstruction changed camera intrinsics and/or poses.")
    else:
        print(
            "  WARNING: Image names/IDs changed, so camera equality could not "
            "be verified without an explicit rename manifest."
        )
    print(f"  Report: {args.report.resolve()}")
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
