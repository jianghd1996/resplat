"""Optional bridge to Reconstruction_full for post-ReSplat Gaussian refinement."""
from __future__ import annotations

import importlib
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _qvec_to_rot(q):
    qw, qx, qy, qz = map(float, q)
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _rot_to_qvec(rot):
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return float(w), float(x), float(y), float(z)


def _normalize_pose(qvec, tvec, world_to_model):
    w2c = np.eye(4)
    w2c[:3, :3] = _qvec_to_rot(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64)
    c2w_model = world_to_model @ np.linalg.inv(w2c)
    w2c_model = np.linalg.inv(c2w_model)
    return _rot_to_qvec(w2c_model[:3, :3]), w2c_model[:3, 3]


def _read_cstr(fp):
    out = bytearray()
    while True:
        value = fp.read(1)
        if not value:
            raise EOFError("Truncated COLMAP images file")
        if value == b"\0":
            return bytes(out)
        out.extend(value)


def _transform_images_bin(path, world_to_model):
    temp = path.with_suffix(path.suffix + ".tmp")
    with path.open("rb") as src, temp.open("wb") as dst:
        (count,) = struct.unpack("<Q", src.read(8))
        dst.write(struct.pack("<Q", count))
        for _ in range(count):
            header = src.read(64)
            if len(header) != 64:
                raise EOFError(f"Truncated COLMAP image header: {path}")
            values = struct.unpack("<i7di", header)
            image_id = values[0]
            qvec, tvec, camera_id = values[1:5], values[5:8], values[8]
            qnew, tnew = _normalize_pose(qvec, tvec, world_to_model)
            dst.write(struct.pack("<i7di", image_id, *qnew, *tnew, camera_id))
            dst.write(_read_cstr(src))
            dst.write(b"\0")
            count_bytes = src.read(8)
            if len(count_bytes) != 8:
                raise EOFError(f"Truncated COLMAP point count: {path}")
            (num_points,) = struct.unpack("<Q", count_bytes)
            dst.write(count_bytes)
            observations = src.read(num_points * 24)
            if len(observations) != num_points * 24:
                raise EOFError(f"Truncated COLMAP observations: {path}")
            dst.write(observations)
    temp.replace(path)


def _transform_points_bin(path, world_to_model):
    temp = path.with_suffix(path.suffix + ".tmp")
    fixed_fmt = "<QdddBBBdQ"
    fixed_size = struct.calcsize(fixed_fmt)
    with path.open("rb") as src, temp.open("wb") as dst:
        count_bytes = src.read(8)
        if len(count_bytes) != 8:
            raise EOFError(f"Truncated COLMAP points3D file: {path}")
        (count,) = struct.unpack("<Q", count_bytes)
        dst.write(count_bytes)
        for _ in range(count):
            fixed = src.read(fixed_size)
            if len(fixed) != fixed_size:
                raise EOFError(f"Truncated COLMAP 3D point: {path}")
            pid, x, y, z, red, green, blue, error, track_count = struct.unpack(fixed_fmt, fixed)
            xyz = world_to_model[:3, :3] @ np.array([x, y, z]) + world_to_model[:3, 3]
            dst.write(struct.pack(fixed_fmt, pid, *map(float, xyz), red, green, blue, error, track_count))
            track = src.read(track_count * 8)
            if len(track) != track_count * 8:
                raise EOFError(f"Truncated COLMAP track: {path}")
            dst.write(track)
    temp.replace(path)


def _transform_images_txt(path, world_to_model):
    lines = path.read_text(encoding="utf-8").splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image record: {line}")
        image_id = int(fields[0])
        qnew, tnew = _normalize_pose(
            list(map(float, fields[1:5])),
            list(map(float, fields[5:8])),
            world_to_model,
        )
        camera_id, name = fields[8], " ".join(fields[9:])
        output.append(
            f"{image_id} " + " ".join(f"{v:.17g}" for v in (*qnew, *tnew))
            + f" {camera_id} {name}"
        )
        if i < len(lines):
            output.append(lines[i])
            i += 1
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _transform_points_txt(path, world_to_model):
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            output.append(line)
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Malformed COLMAP point record: {line}")
        xyz = world_to_model[:3, :3] @ np.asarray(list(map(float, fields[1:4]))) + world_to_model[:3, 3]
        fields[1:4] = [f"{v:.17g}" for v in xyz]
        output.append(" ".join(fields))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _transform_colmap(sparse_dir, reference_c2w):
    reference = np.asarray(reference_c2w, dtype=np.float64)
    if reference.shape != (4, 4):
        raise ValueError(f"Expected 4x4 reference camera pose, got {reference.shape}")
    world_to_model = np.linalg.inv(reference)
    images_bin, images_txt = sparse_dir / "images.bin", sparse_dir / "images.txt"
    points_bin, points_txt = sparse_dir / "points3D.bin", sparse_dir / "points3D.txt"
    if images_bin.is_file():
        _transform_images_bin(images_bin, world_to_model)
    if images_txt.is_file():
        _transform_images_txt(images_txt, world_to_model)
    if points_bin.is_file():
        _transform_points_bin(points_bin, world_to_model)
    if points_txt.is_file():
        _transform_points_txt(points_txt, world_to_model)
    if not images_bin.is_file() and not images_txt.is_file():
        raise FileNotFoundError(f"COLMAP images.bin/txt missing at {sparse_dir}")


def _return_plys(value, search_roots):
    found = []
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [
            Path(root) / path for root in search_roots
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".ply":
                found.append(candidate.resolve())
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_return_plys(item, search_roots))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.extend(_return_plys(item, search_roots))
    return found


def _iteration(path):
    numbers = re.findall(r"iteration[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(numbers[-1]) if numbers else -1


def _find_output_ply(result, export_dir, work_dir, excluded=()):
    search_roots = (export_dir, work_dir)
    returned = _return_plys(result, search_roots)
    excluded = {Path(path).resolve() for path in excluded}
    candidates = list(returned)
    for root in search_roots:
        candidates.extend(p.resolve() for p in root.rglob("*.ply") if p.is_file())
    candidates = list(dict.fromkeys(
        path for path in candidates if path not in excluded
    ))
    if not candidates:
        raise FileNotFoundError(
            "Reconstruction_full returned no PLY and none was found under "
            f"{export_dir} or {work_dir}"
        )
    selected = max(
        candidates,
        key=lambda p: (
            _iteration(p),
            p in returned,
            p.stat().st_mtime_ns,
            p.stat().st_size,
        ),
    )
    return selected, selected in returned


def _clear(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def run_reconstruction_full(
    *, initial_ply, scene_path, images_dir, sparse_dir, work_dir, export_dir,
    output_ply, code_root, reference_c2w_original, mode="scene", circle=0,
    max_frames=60,
):
    initial_ply = Path(initial_ply).resolve()
    scene_path = Path(scene_path).resolve()
    work_dir, export_dir, output_ply = map(Path, (work_dir, export_dir, output_ply))
    code_root = Path(code_root).expanduser().resolve()
    image_source = (scene_path / images_dir).resolve()
    sparse_source = (scene_path / sparse_dir).resolve()
    for path in (initial_ply, image_source, sparse_source, code_root):
        if not path.exists():
            raise FileNotFoundError(path)

    _clear(work_dir)
    _clear(export_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "image").symlink_to(image_source, target_is_directory=True)
    sparse_stage = work_dir / "Sparse" / "0"
    sparse_stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sparse_source, sparse_stage)
    _transform_colmap(sparse_stage, reference_c2w_original)
    # Reconstruction_full's dense-init convention uses this PLY as its seed.
    staged_seed_ply = sparse_stage / "points3D.ply"
    shutil.copy2(initial_ply, staged_seed_ply)
    # Support both the case used in the supplied call and the lowercase path
    # used by scripts/reconstruction in VideoX-Fun.
    (work_dir / "sparse").symlink_to("Sparse", target_is_directory=True)

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    importlib.invalidate_caches()
    try:
        from Reconstruction_full import Reconstruction
    except ImportError as exc:
        raise ImportError(
            f"Could not import Reconstruction_full from {code_root}; "
            "set --reconstruction-code-root to the folder's parent."
        ) from exc

    load_start = time.perf_counter()
    recon = Reconstruction(preload=True, dense_init_scene=False)
    preload_seconds = time.perf_counter() - load_start
    print(f"[MODULE B] Reconstruction_full preload {preload_seconds:.3f}s", flush=True)

    run_start = time.perf_counter()
    result = recon(
        work_dir=work_dir, export_dir=export_dir, mode=mode,
        circle=circle, max_frames=max_frames,
    )
    optimize_seconds = time.perf_counter() - run_start
    selected, returned_path = _find_output_ply(
        result, export_dir, work_dir, excluded=(staged_seed_ply,)
    )
    with selected.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Reconstruction_full output is not a PLY: {selected}")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, output_ply)
    print(f"[MODULE B] Optimized PLY: {selected}", flush=True)
    return output_ply, {
        "name": "Reconstruction_full",
        "is_quality_optimization": True,
        "input_ply": str(initial_ply),
        "output_ply": str(output_ply),
        "optimizer_output_ply": str(selected),
        "optimizer_output_selected_from_return_value": returned_path,
        "work_dir": str(work_dir),
        "export_dir": str(export_dir),
        "mode": mode,
        "circle": circle,
        "max_frames": max_frames,
        "preload_seconds_included_in_module_b": round(preload_seconds, 6),
        "optimization_seconds": round(optimize_seconds, 6),
        "returned_result_type": type(result).__name__,
        "returned_result_keys": list(result.keys()) if isinstance(result, dict) else None,
        "output_matches_initial_ply": initial_ply.read_bytes() == output_ply.read_bytes(),
    }

