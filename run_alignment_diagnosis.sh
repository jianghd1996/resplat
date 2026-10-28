#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE="${SCENE:-$REPO_ROOT/datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/competition/demo_reconstruction_full}"

cd "$REPO_ROOT"

COMMAND=(
  "$PYTHON_BIN" scripts/diagnose_reconstruction_alignment.py
  --scene "$SCENE"
  --pipeline-output "$RUN_DIR"
)
if [[ -n "${COLMAP_FINAL:-}" \
      && -f "$COLMAP_FINAL/cameras.txt" \
      && -f "$COLMAP_FINAL/images.txt" ]]; then
  COMMAND+=(--colmap-final "$COLMAP_FINAL")
elif [[ -n "${COLMAP_FINAL:-}" ]]; then
  echo "Ignoring stale COLMAP_FINAL: $COLMAP_FINAL" >&2
  echo "Auto-selecting the latest complete result_gs/*/colmap_final." >&2
fi

"${COMMAND[@]}"
