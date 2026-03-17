# ODE Regression Training Guide

## Overview

This document summarizes the ODE regression training path we connected into `ltx-trainer`.

The full pipeline is:

1. Generate ODE trajectory `.pt` samples
2. Convert trajectory samples into `.precomputed` format for `ltx-trainer`
3. Use the new `ode_regression` training strategy to train the LTX transformer

The training target is still expressed in the LTX model's native prediction space:

- current state: `x_t`
- clean target: `x0`
- sigma: `sigma`
- velocity target used for training: `(x_t - x0) / sigma`

This keeps the trainer compatible with the existing LTX transformer and validation code.

## Main Files

### Data generation and conversion

- `myltx/ode/gen_ode_data.py`
  - Generates original ODE trajectory `.pt` files
  - Output sample structure includes:
    - `stage1_video_traj`
    - `stage1_audio_traj`
    - `stage1_sigmas`
    - `stage2_video_traj`
    - `stage2_audio_traj`
    - `stage2_sigmas`
    - `noise_seeds`
      - deterministic per-sample noise metadata
      - stage1/stage2 and video/audio use explicitly derived independent seeds

- `myltx/ode/configs/gen_ode_data.yaml`
  - Non-distilled two-stage ODE data generation config

- `myltx/ode/configs/gen_ode_data_distilled.yaml`
  - Distilled ODE data generation config
  - Also provides defaults used by the conversion script

- `myltx/ode/convert_ode_pt_to_precomputed.py`
  - Converts ODE trajectory `.pt` files into `ltx-trainer` `.precomputed` layout
  - Supports:
    - `--export-mode standard`
    - `--export-mode ode_regression`
    - `--trajectory-step first`
    - `--trajectory-step last`
    - `--trajectory-step all`
    - `--trajectory-step all_non_last`
  - In `ode_regression` mode it writes:
    - `latents`
    - `ode_target_latents`
    - `ode_sigma`
    - `ode_step_index`
    - `ode_clean_step_index`
    - `ode_noise_seeds` when available in the source trajectory sample

### Trainer integration

- `myltx/packages/ltx-trainer/src/ltx_trainer/training_strategies/ode_regression.py`
  - New ODE regression training strategy
  - Reads current latent, clean latent target, and sigma
  - Converts clean target to velocity target
  - Supports joint audio-video loss

- `myltx/packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py`
  - Registers `ODERegressionStrategy`

- `myltx/packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py`
  - Base strategy discriminator extended with `ode_regression`

- `myltx/packages/ltx-trainer/src/ltx_trainer/config.py`
  - Adds `ODERegressionConfig` to the training strategy config union

- `myltx/packages/ltx-trainer/src/ltx_trainer/datasets.py`
  - Loads `.precomputed` data
  - Normalizes:
    - video latents separately
    - audio latents separately
  - Important fix:
    - `audio_latents` must not be normalized with video metadata like `num_frames`

- `myltx/packages/ltx-trainer/scripts/train.py`
  - Training entry point

### Training configs

- `myltx/packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml`
  - Main ODE regression training config template

- `myltx/packages/ltx-trainer/configs/ltx2_av_ode_regression.test.yaml`
  - Smoke-test config
  - Uses:
    - local model paths
    - local Gemma path
    - local converted test dataset
    - `optimization.steps: 2`

### Reference implementation from causAV

- `causAV/causav/train_ode.py`
  - Reference training entry point

- `causAV/causav/trainer/ode.py`
  - Reference trainer loop

- `causAV/causav/model/ode_regression.py`
  - Reference ODE regression objective
  - Important conceptual reference:
    - use current trajectory state as input
    - use clean latent as supervision target

## Data Layout

After conversion, the expected dataset layout is:

```text
<dataset_root>/
├── .precomputed/
│   ├── latents/
│   ├── audio_latents/
│   └── conditions/
└── conversion_manifest.json
```

Each `latents/*.pt` sample in `ode_regression` mode contains fields like:

```python
{
    "latents": ...,
    "num_frames": ...,
    "height": ...,
    "width": ...,
    "fps": ...,
    "ode_target_latents": ...,
    "ode_sigma": ...,
    "ode_step_index": ...,
    "ode_clean_step_index": ...,
}
```

Each `audio_latents/*.pt` sample contains:

```python
{
    "latents": ...,
    "num_time_steps": ...,
    "frequency_bins": ...,
    "duration": ...,
    "ode_target_latents": ...,
    "ode_sigma": ...,
    "ode_step_index": ...,
    "ode_clean_step_index": ...,
}
```

`conditions/*.pt` contains text condition embeddings.

## Recommended Commands

### 1. Activate environment

Always run from `myltx`:

```bash
cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx
source .venv/bin/activate
```

You can then use either `uv run ...` or plain `python ...`.
This repo generally prefers `uv run`.

### 2. Export a small ODE regression test dataset

```bash
uv run python ode/convert_ode_pt_to_precomputed.py \
  --config ode/configs/gen_ode_data_distilled.yaml \
  --input-dir ode/data_distilled \
  --export-mode ode_regression \
  --trajectory-step all_non_last \
  --limit 2 \
  --output-dir ode/data_distilled_stage2_ode_test
```

Notes:

- `--limit 2` means only the first 2 original `.pt` samples are converted
- `--trajectory-step all_non_last` expands each trajectory into multiple noisy-to-clean training items
- This is the recommended mode for ODE regression training

### 3. Run smoke-test training

```bash
uv run python packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/ltx2_av_ode_regression.test.yaml
```

This config is already prepared to use:

- model: `myltx/model/ltx-2.3-22b-distilled.safetensors`
- text encoder: `myltx/model/gemma`
- dataset: `myltx/ode/data_distilled_stage2_ode_test`

### 4. Run full training

First update:

- `packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml`

Then run:

```bash
uv run python packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml
```

For multi-GPU training:

```bash
uv run accelerate launch packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml
```

## Important Config Parameters

### Conversion script parameters

In `convert_ode_pt_to_precomputed.py`:

- `--input-dir`
  - Directory containing original ODE trajectory `.pt` files

- `--output-dir`
  - Output dataset root

- `--stage`
  - `stage1` or `stage2`
  - Usually `stage2` is used for current LTX ODE regression training

- `--trajectory-step`
  - `first`: only first step
  - `last`: only clean step
  - `all`: every step including clean
  - `all_non_last`: all noisy/intermediate steps, excluding clean
  - Recommended for ODE regression: `all_non_last`

- `--export-mode`
  - `standard`: plain latent export
  - `ode_regression`: writes target latent and sigma supervision
  - Required for ODE regression training

- `--limit`
  - Useful for smoke tests

- `--no-write-audio`
  - If set, disables audio export

- `--no-write-conditions`
  - If set, disables condition export

### Training config parameters

In `ltx2_av_ode_regression.yaml`:

#### `model`

- `model.model_path`
  - Base LTX checkpoint path

- `model.text_encoder_path`
  - Gemma directory path

- `model.training_mode`
  - Usually `lora`

#### `training_strategy`

- `training_strategy.name`
  - Must be `ode_regression`

- `training_strategy.with_audio`
  - `true`: joint AV training
  - `false`: video-only ODE regression

- `training_strategy.audio_latents_dir`
  - Directory name under `.precomputed`
  - Usually `audio_latents`

- `training_strategy.sigma_epsilon`
  - Numerical floor for sigma
  - Samples with sigma below this are effectively ignored in loss

- `training_strategy.loss_reweight_mode`
  - `manual`
  - `auto`

- `training_strategy.video_loss_weight`
  - Video loss weight in manual mode

- `training_strategy.audio_loss_weight`
  - Audio loss weight in manual mode

#### `optimization`

- `optimization.steps`
  - Number of training steps

- `optimization.batch_size`
  - Per-device batch size

- `optimization.gradient_accumulation_steps`
  - Gradient accumulation

- `optimization.learning_rate`
  - Optimizer LR

- `optimization.enable_gradient_checkpointing`
  - Important for memory reduction

#### `acceleration`

- `acceleration.mixed_precision_mode`
  - Usually `bf16`

- `acceleration.quantization`
  - Optional

- `acceleration.load_text_encoder_in_8bit`
  - Reduce text encoder memory if needed

#### `data`

- `data.preprocessed_data_root`
  - Must point to the converted ODE dataset root
  - Can be either dataset root or `.precomputed` root

- `data.num_dataloader_workers`
  - Number of workers

#### `validation`

- `validation.interval`
  - Set `null` to disable validation during smoke tests

- `validation.prompts`
  - Validation prompts

- `validation.generate_audio`
  - Whether to generate audio during validation

## Recommended First Runs

### Smoke test

Use:

- conversion with `--limit 2`
- training config `ltx2_av_ode_regression.test.yaml`
- `optimization.steps: 2`

This verifies:

- dataset can be loaded
- strategy can prepare ODE targets
- model can complete forward/backward
- checkpoint can be saved

### Full run

For real training:

1. Export the full dataset
2. Point `ltx2_av_ode_regression.yaml` to the full dataset root
3. Increase:
   - `optimization.steps`
   - optionally batch size
4. Enable multi-GPU with `accelerate launch` if needed

## Known Issue That Was Fixed

The first runtime error we hit was:

- `audio_latents` was incorrectly normalized as video latents
- `datasets.py` tried to read `data["num_frames"]` from an audio payload
- audio payload only has:
  - `num_time_steps`
  - `frequency_bins`

Fix:

- `datasets.py` now normalizes:
  - video latents with `_normalize_video_latents()`
  - audio latents with `_normalize_audio_latents()`

## Current Status

The smoke test has already run successfully end-to-end with:

- converted test ODE dataset
- `ltx2_av_ode_regression.test.yaml`
- 2 training steps

Artifacts produced:

- output dir: `outputs/ode_regression_smoke_test`
- checkpoint: `checkpoints/lora_weights_step_00002.safetensors`

## Minimal Ready-to-Run Commands

```bash
cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx
source .venv/bin/activate
```

```bash
uv run python ode/convert_ode_pt_to_precomputed.py \
  --config ode/configs/gen_ode_data_distilled.yaml \
  --input-dir ode/data_distilled \
  --export-mode ode_regression \
  --trajectory-step all_non_last \
  --limit 2 \
  --output-dir ode/data_distilled_stage2_ode_test
```

```bash
uv run python packages/ltx-trainer/scripts/train.py \
  packages/ltx-trainer/configs/ltx2_av_ode_regression.test.yaml
```
