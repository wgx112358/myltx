#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$ROOT_DIR/log"

jobs=(
  "decode_gt_spot.sbatch"
  "exact_replay_spot.sbatch"
  "official_distilled_spot.sbatch"
  "ode_pipe_distilled_spot.sbatch"
  "ode_base_step250_spot.sbatch"
  "ode_latest_step1000_spot.sbatch"
  "ode_run1_step1000_spot.sbatch"
)

mkdir -p "$LOG_ROOT"

for script in "${jobs[@]}"; do
  echo "[submit] sbatch $script"
  sbatch "$ROOT_DIR/$script"
done
