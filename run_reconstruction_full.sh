#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"
RECONSTRUCTION_CODE_ROOT="${RECONSTRUCTION_CODE_ROOT:-$REPO_ROOT}"
RECONSTRUCTION_MODE="${RECONSTRUCTION_MODE:-scene}"
RECONSTRUCTION_CIRCLE="${RECONSTRUCTION_CIRCLE:-0}"
RECONSTRUCTION_MAX_FRAMES="${RECONSTRUCTION_MAX_FRAMES:-60}"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/run_reconstruction_full_stage.py \
  --scene "$SCENE" \
  --pipeline-output "$RUN_DIR" \
  --reconstruction-code-root "$RECONSTRUCTION_CODE_ROOT" \
  --mode "$RECONSTRUCTION_MODE" \
  --circle "$RECONSTRUCTION_CIRCLE" \
  --max-frames "$RECONSTRUCTION_MAX_FRAMES"

echo
echo "Reconstruction_full complete:"
echo "  $RUN_DIR/module_b/gaussians.ply"
echo "  $RUN_DIR/module_b/reconstruction_full_report.json"
