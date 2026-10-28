#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"
BASELINE_PLY="${BASELINE_PLY:-$RUN_DIR/resplat/gaussians.ply}"
OPTIMIZED_PLY="${OPTIMIZED_PLY:-$RUN_DIR/module_b/gaussians.ply}"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/evaluate_ply_comparison.py \
  --scene "$SCENE" \
  --pipeline-output "$RUN_DIR" \
  --baseline-ply "$BASELINE_PLY" \
  --optimized-ply "$OPTIMIZED_PLY" \
  --device "$DEVICE"

echo
echo "Comparison complete:"
echo "  $RUN_DIR/comparison/comparison_report.json"
