#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
START_FRAME="${START_FRAME:-0}"
FRAME_DISTANCE="${FRAME_DISTANCE:-60}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_${FRAME_DISTANCE}}"
EVALUATION_COUNT="${EVALUATION_COUNT:-20}"
EVALUATION_SEED="${EVALUATION_SEED:-2026}"

# Direct refinement settings. Any caller-provided environment values win.
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

export PYTHON_BIN DEVICE SCENE START_FRAME FRAME_DISTANCE RUN_DIR
export EVALUATION_COUNT EVALUATION_SEED
export REFINE_STEPS REFINE_BATCH_SIZE REFINE_VIEWS_PER_UPDATE
export REFINE_MODE REFINE_SEED
export REFINE_LR_SH REFINE_LR_OPACITY REFINE_LR_MEANS
export REFINE_LR_SCALES REFINE_LR_QUATS

cd "$REPO_ROOT"

echo "============================================================"
echo "Direct ReSplat competition pipeline"
echo "  Frame window: [$START_FRAME, $((START_FRAME + FRAME_DISTANCE)))"
echo "  Output:       $RUN_DIR"
echo "  Evaluation:   $EVALUATION_COUNT held-out views (seed=$EVALUATION_SEED)"
echo "  Refinement:   $REFINE_MODE, $REFINE_STEPS updates,"
echo "                $REFINE_VIEWS_PER_UPDATE views/update"
echo "============================================================"

echo
echo "[1/3] ReSplat export and fixed-view baseline evaluation"
bash "$REPO_ROOT/run_resplat_export.sh"

echo
echo "[2/3] Direct few-step gsplat refinement"
bash "$REPO_ROOT/run_resplat_refinement.sh"

echo
echo "[3/3] Same-view native-resolution comparison"
bash "$REPO_ROOT/run_gaussian_comparison.sh"

echo
echo "============================================================"
echo "Pipeline complete"
echo "  Baseline:   $RUN_DIR/resplat/gaussians_model_frame.ply"
echo "  Optimized:  $RUN_DIR/module_b/gaussians.ply"
echo "  Comparison: $RUN_DIR/comparison/comparison_report.json"
echo "============================================================"
