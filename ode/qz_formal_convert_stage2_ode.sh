#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"

cd "$ROOT"

export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"

python ode/convert_ode_pt_to_precomputed.py \
  --input-dir ode/data_distilled \
  --output-dir ode/data_distilled_stage2_ode \
  --stage stage2 \
  --export-mode ode_regression \
  --trajectory-step all_non_last
