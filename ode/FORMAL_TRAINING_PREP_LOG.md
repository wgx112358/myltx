# Formal Training Prep Log

## 2026-03-17 07:30:33 UTC

- Request: prepare the repository for formal ODE regression training.
- Verified that the interrupted conversion left a partial dataset under `ode/data_distilled_stage2_ode/`.
- Verified block-causal defaults in `packages/ltx-trainer/src/ltx_trainer/training_strategies/ode_regression.py`:
  - `block_size = 6`
  - `independent_first_frame = true`
  - `audio_boundary_mode = "left"`
- Updated `packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml`:
  - set `data.preprocessed_data_root` to `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled_stage2_ode`
  - explicitly set:
    - `use_block_causal_mask: true`
    - `block_size: 6`
    - `independent_first_frame: true`
    - `audio_boundary_mode: "left"`
- Cleaned smoke/test artifacts created during previous verification:
  - `outputs/ode_e2e_autosmoke`
  - `outputs/ode_regression_blockmask_integration_smoke`
  - `outputs/ode_regression_blockmask_integration_smoke.yaml`
  - `outputs/ode_regression_smoke_test`
  - `outputs/ode_regression_wandb_smoke`
  - `outputs/ode_regression_wandb_smoke_config.yaml`
  - `outputs/ode_regression_wandb_online_smoke`
  - `outputs/ode_regression_wandb_online_smoke_config.yaml`
  - `ode/data_distilled_stage2_ode_test`
  - `wandb/`

## Next Planned Action

- Remove the partial `ode/data_distilled_stage2_ode/` directory created by the interrupted conversion.
- Re-run the formal conversion from `ode/data_distilled/` into `ode/data_distilled_stage2_ode/` using:
  - `--stage stage2`
  - `--export-mode ode_regression`
  - `--trajectory-step all_non_last`

## 2026-03-17 07:32:27 UTC

- Executed cleanup of the interrupted formal dataset output:
  - removed `ode/data_distilled_stage2_ode/`
- Ready to restart the formal conversion from scratch.

## 2026-03-17 07:33:13 UTC

- Restarted the formal conversion command from scratch:
  - `source .venv/bin/activate`
  - `PYTHONPATH=packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src`
  - `python ode/convert_ode_pt_to_precomputed.py --input-dir ode/data_distilled --output-dir ode/data_distilled_stage2_ode --stage stage2 --export-mode ode_regression --trajectory-step all_non_last`
- Observed startup status:
  - discovered `12000` raw ODE `.pt` files
  - using `stage2`
  - using `export_mode=ode_regression`
  - using `trajectory_step=all_non_last`
  - text encoder path resolved to `model/gemma`

## 2026-03-17 07:37:54 UTC

- Switched the long-running formal conversion from an interactive foreground process to a persistent `tmux` session.
- Cleaned the output directory again before re-launching:
  - removed `ode/data_distilled_stage2_ode/`
- Re-launched the exact same conversion command in:
  - tmux session: `formal_ode_convert`
- Persistent log file:
  - `ode/logs/formal_convert_stage2_ode.log`
- Verified the session is alive and writing progress into the log.

## 2026-03-17 07:47:02 UTC

- User requested moving the slow formal conversion to the QZ platform with low priority and 8 GPUs.
- Stopped the local `tmux` conversion session:
  - `formal_ode_convert`
- Added QZ submission script:
  - `myltx/ode/qz_formal_convert_stage2_ode.sh`
- Submitted QZ job with:
  - machine: `h200-2`
  - GPUs: `8`
  - priority: `p3`
  - conda env: `diffsynth`
  - version: `v1`
  - experiment: `formal-ode-convert-stage2`
- QZ job result:
  - `job_id = job-dfad6d21-7164-4402-8e8f-b5e8df45d4d6`
  - `name = wgx-train-h200-8g-p3-v1-formal-ode-convert-stage2`
  - `status = job_queuing`
- QZ log path:
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-8g-p3-v1-formal-ode-convert-stage2.log`

## 2026-03-17 07:51:40 UTC

- User corrected the requirement:
  - use `8-way sharded parallel conversion`
  - use `H200-3号机房`
- Stopped the previous non-sharded QZ job:
  - `job-dfad6d21-7164-4402-8e8f-b5e8df45d4d6`
- Added sharded QZ conversion script:
  - `myltx/ode/qz_formal_convert_stage2_ode_sharded.sh`
- Sharding design:
  - split sorted `ode/data_distilled/*.pt` evenly by file index modulo `8`
  - create `8` shard input directories via symlinks
  - launch `8` parallel converter processes
  - assign one visible GPU per shard via `CUDA_VISIBLE_DEVICES=<shard_id>`
  - merge shard outputs back into:
    - `ode/data_distilled_stage2_ode/.precomputed/{latents,audio_latents,conditions}`
  - write a merged `conversion_manifest.json`
- Submitted the corrected QZ job:
  - `job_id = job-7c360c4a-ce9b-4bd1-a938-58285e3e0c34`
  - `name = wgx-train-h200-8g-p3-v2-formal-ode-convert-stage2-sharded`
  - `machine = H200-3号机房`
  - `priority = p3`
  - `status = job_creating`
- Corrected QZ log path:
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-8g-p3-v2-formal-ode-convert-stage2-sharded.log`

## 2026-03-17 07:55:06 UTC

- User requested terminating the current QZ task and submitting again.
- Stopped:
  - `job-7c360c4a-ce9b-4bd1-a938-58285e3e0c34`
  - `wgx-train-h200-8g-p3-v2-formal-ode-convert-stage2-sharded`
- Re-submitted the same 8-way sharded parallel conversion configuration with a new task version:
  - `version = v3`
  - `machine = H200-3号机房`
  - `gpus = 8`
  - `priority = p3`
- New QZ job:
  - `job_id = job-6ba808a7-2009-4a45-af06-522e01fed7c3`
  - `name = wgx-train-h200-8g-p3-v3-formal-ode-convert-stage2-sharded`
  - `status = job_creating`
- New QZ log path:
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-8g-p3-v3-formal-ode-convert-stage2-sharded.log`

## 2026-03-17 07:58:44 UTC

- User requested another submission on `H200-3号机房-2`.
- Kept the existing `v3` task intact and submitted an additional task instead of replacing it.
- New QZ job:
  - `job_id = job-385c8f8c-c988-4475-8e23-acba11fdac1f`
  - `name = wgx-train-h200-8g-p3-v4-formal-ode-convert-stage2-sharded`
  - `machine = H200-3号机房-2`
  - `gpus = 8`
  - `priority = p3`
  - `status = job_queuing`
- New QZ log path:
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-8g-p3-v4-formal-ode-convert-stage2-sharded.log`

## 2026-03-17 08:08:23 UTC

- User reported that the previous QZ command used the wrong environment (`conda activate diffsynth`).
- Corrected the sharded conversion script to use the uv environment directly:
  - hard-coded python path: `myltx/.venv/bin/python`
  - removed reliance on shell `python`
- Corrected the QZ submission command to disable conda activation:
  - `--conda-env ""`
- Submitted corrected UV-based task to `H200-3号机房`:
  - `job_id = job-be48cae1-484c-4290-ab04-94dc7993cb9f`
  - `name = wgx-train-h200-8g-p3-v5-formal-ode-convert-stage2-sharded-uv`
  - `machine = H200-3号机房`
  - `gpus = 8`
  - `priority = p3`
  - `status = job_creating`
- Corrected QZ log path:
  - `/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/qz/logs/wgx-train-h200-8g-p3-v5-formal-ode-convert-stage2-sharded-uv.log`
