"""External Module B interface.

The first committed implementation is an identity baseline. It preserves the
ReSplat PLY byte-for-byte so the complete evaluation and timing path can be
validated before an optimization algorithm is plugged in.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_module_b(source_ply: Path, output_ply: Path, method: str = "identity"):
    source_ply = Path(source_ply)
    output_ply = Path(output_ply)
    if method != "identity":
        raise ValueError(f"Unsupported Module B method: {method}")
    if not source_ply.is_file():
        raise FileNotFoundError(source_ply)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ply, output_ply)
    source_hash = sha256(source_ply)
    output_hash = sha256(output_ply)
    if source_hash != output_hash:
        raise RuntimeError("Identity Module B output differs from the input PLY.")

    return output_ply, {
        "name": "identity_baseline",
        "is_quality_optimization": False,
        "input_ply": str(source_ply),
        "output_ply": str(output_ply),
        "input_sha256": source_hash,
        "output_sha256": output_hash,
        "gaussian_count_changed": False,
    }
