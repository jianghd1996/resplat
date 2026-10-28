#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"

cd "$REPO_ROOT"

"$PYTHON_BIN" run_pipeline.py \
  --scene "$SCENE" \
  --output "$RUN_DIR" \
  --module-b identity \
  --export-reconstruction-init \
  --device "$DEVICE"

echo
echo "ReSplat export complete:"
echo "  $RUN_DIR/resplat/gaussians.ply"
echo "  $RUN_DIR/resplat/gaussians_model_frame.ply"
echo "  $RUN_DIR/resplat/reconstruction_frame.json"
