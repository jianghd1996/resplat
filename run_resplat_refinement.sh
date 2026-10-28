#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"
INPUT_PLY="${INPUT_PLY:-$RUN_DIR/resplat/gaussians_model_frame.ply}"
OUTPUT_PLY="${OUTPUT_PLY:-$RUN_DIR/module_b/gaussians.ply}"
REFINE_STEPS="${REFINE_STEPS:-8}"
REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-2}"
REFINE_VIEWS_PER_UPDATE="${REFINE_VIEWS_PER_UPDATE:-8}"
REFINE_MODE="${REFINE_MODE:-geometry_lite}"
REFINE_SEED="${REFINE_SEED:-0}"
REFINE_LR_SH="${REFINE_LR_SH:-2.5e-4}"
REFINE_LR_OPACITY="${REFINE_LR_OPACITY:-1.0e-4}"
REFINE_LR_MEANS="${REFINE_LR_MEANS:-2.0e-7}"
REFINE_LR_SCALES="${REFINE_LR_SCALES:-2.0e-6}"
REFINE_LR_QUATS="${REFINE_LR_QUATS:-1.0e-6}"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/refine_resplat_gaussians.py \
  --scene "$SCENE" \
  --pipeline-output "$RUN_DIR" \
  --input-ply "$INPUT_PLY" \
  --output-ply "$OUTPUT_PLY" \
  --device "$DEVICE" \
  --steps "$REFINE_STEPS" \
  --batch-size "$REFINE_BATCH_SIZE" \
  --views-per-update "$REFINE_VIEWS_PER_UPDATE" \
  --mode "$REFINE_MODE" \
  --seed "$REFINE_SEED" \
  --lr-sh "$REFINE_LR_SH" \
  --lr-opacity "$REFINE_LR_OPACITY" \
  --lr-means "$REFINE_LR_MEANS" \
  --lr-scales "$REFINE_LR_SCALES" \
  --lr-quats "$REFINE_LR_QUATS"

echo
echo "Direct refinement complete:"
echo "  $OUTPUT_PLY"
echo "  $RUN_DIR/module_b/direct_refinement_report.json"
echo
echo "Evaluate with:"
echo "  bash run_gaussian_comparison.sh"
