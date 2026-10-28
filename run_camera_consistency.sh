#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"
INPUT_SPARSE="${INPUT_SPARSE:-$RUN_DIR/module_b/reconstruction_work/sparse/0}"

if [[ -z "${COLMAP_FINAL:-}" \
      || ! -f "$COLMAP_FINAL/cameras.txt" \
      || ! -f "$COLMAP_FINAL/images.txt" ]]; then
  if [[ -n "${COLMAP_FINAL:-}" ]]; then
    echo "Ignoring stale COLMAP_FINAL: $COLMAP_FINAL" >&2
  fi
  COLMAP_FINAL="$(find "$RUN_DIR/module_b/reconstruction_work/result_gs" \
    -type f -path '*/colmap_final/cameras.txt' -printf '%T@ %h\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$COLMAP_FINAL" ]]; then
  echo "No colmap_final directory found." >&2
  exit 1
fi

REPORT="${CAMERA_REPORT:-$RUN_DIR/comparison/camera_consistency_report.json}"

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/compare_colmap_cameras.py \
  --input-sparse "$INPUT_SPARSE" \
  --output-colmap "$COLMAP_FINAL" \
  --report "$REPORT"
