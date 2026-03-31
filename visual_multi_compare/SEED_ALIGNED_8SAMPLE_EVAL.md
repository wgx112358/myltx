# Seed-Aligned 8-Sample Eval

## 1. 文档路径

本文档路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/SEED_ALIGNED_8SAMPLE_EVAL.md`

## 2. 这次实现了什么

这次补齐的是一套用于 **8 个样本、按 precompute 噪声 seed 对齐** 的对比评估脚本。

核心目标有两点：

1. 不再让批量推理统一使用固定 `--seed 42`
2. 改为让每个样本使用其 precompute `.pt` 中记录的 `ode_noise_seeds.stage2.video`

这样做的意义是：

- 推理初始化噪声和 GT / precompute 轨迹保持一致
- ODE 训练效果变好时，输出会更接近对应 GT
- 视觉对比才有因果意义，而不是被 seed 偏差污染

## 3. 涉及的程序与路径

### 3.1 采样脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/sample_5_indices.py`

作用：

- 从 `datagen/ltx_prompts_v1_12000.csv` 中抽样 8 个 index
- 读取对应 precompute `.pt`
- 提取 `ode_noise_seeds.stage2.video`
- 生成 `sampled_8.json`

输出文件默认路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/sampled_8.json`

生成后的每条 sample 形如：

```json
{
  "index": 1681,
  "prompt": "...",
  "seed": 5534811387362042849,
  "noise_seeds": {
    "scheme": "per-stage-per-modality-v1",
    "base_seed": 42,
    "global_idx": 1681,
    "stage1": {
      "video": 8454862770128345870,
      "audio": 7858755530059690374
    },
    "stage2": {
      "video": 5534811387362042849,
      "audio": 2925523964658678304
    }
  }
}
```

其中真正用于推理对齐的是：

`seed == noise_seeds.stage2.video`

### 3.2 ODE 批量推理脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/infer_5prompt_ode.py`

作用：

- 读取 `sampled_8.json`
- 对每个样本解析自己的 seed
- 调用 `CausalDistilledODEPipeline`
- 输出 ODE 模型的视频结果

seed 解析优先级：

1. `sample["noise_seeds"]["stage2"]["video"]`
2. `sample["seed"]`
3. 命令行 `--seed`

### 3.3 Official distilled 批量推理脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/infer_5prompt_distilled.py`

作用：

- 读取同一个 `sampled_8.json`
- 对每个样本使用相同的 seed 解析逻辑
- 调用官方 `DistilledPipeline`
- 输出官方蒸馏模型的视频结果

seed 解析优先级与 ODE 脚本相同：

1. `sample["noise_seeds"]["stage2"]["video"]`
2. `sample["seed"]`
3. 命令行 `--seed`

### 3.4 GT 解码脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/decode_gt_from_precompute.py`

作用：

- 从 precompute 中读取 GT / clean latent
- 只加载 VAE decoder
- 解码出 GT 视频

这个脚本本次没有改 seed 逻辑，但和 `sampled_8.json` 一起组成完整对比流程。

## 4. 对应 sbatch 路径

所有 sbatch 统一放在 `visual_multi_compare/` 下，使用 `visual_multi_compare/sampled_8.json`：

- `visual_multi_compare/exact_replay_spot.sbatch`
- `visual_multi_compare/official_distilled_spot.sbatch`
- `visual_multi_compare/ode_pipe_distilled_spot.sbatch`
- `visual_multi_compare/ode_base_step250_spot.sbatch`
- `visual_multi_compare/ode_latest_step1000_spot.sbatch`
- `visual_multi_compare/ode_run1_step1000_spot.sbatch`

一键提交：`bash visual_multi_compare/submit_all_spot.sh`

## 5. 用法

### 5.1 先生成 8 个样本清单

在仓库根目录执行：

```bash
.venv/bin/python visual_multi_compare/sample_5_indices.py
```

如果需要指定参数：

```bash
.venv/bin/python visual_multi_compare/sample_5_indices.py \
  --csv-path datagen/ltx_prompts_v1_12000.csv \
  --precompute-root /mnt/petrelfs/wanggongxuan/workspace/ode_precompute_stage2_8gpu_20260324T125612Z/.precomputed \
  --output-path visual_multi_compare/sampled_8.json \
  --sample-seed 2026 \
  --num-samples 8 \
  --total 12000
```

产物：

`visual_multi_compare/sampled_8.json`

### 5.2 跑 ODE 批量推理

手工运行示例：

```bash
.venv/bin/python visual_multi_compare/infer_5prompt_ode.py \
  --samples-json visual_multi_compare/sampled_8.json \
  --ode-checkpoint-path /path/to/ode_checkpoint.safetensors \
  --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root model/gemma \
  --tag ode_eval \
  --output-root /path/to/output_root
```

说明：

- 即使传了 `--seed`，只要 sample 里存在 `noise_seeds.stage2.video` 或 `seed`，脚本都会优先使用 sample 自己的 seed
- `--seed` 主要用于兼容旧格式 sample JSON

输出目录形如：

```text
/path/to/output_root/ode_eval/
  ode_eval_01681.mp4
  ode_eval_01951.mp4
  ...
```

### 5.3 跑 official distilled 批量推理

```bash
.venv/bin/python visual_multi_compare/infer_5prompt_distilled.py \
  --samples-json visual_multi_compare/sampled_8.json \
  --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root model/gemma \
  --tag official_distilled \
  --output-root /path/to/output_root
```

输出目录形如：

```text
/path/to/output_root/official_distilled/
  official_distilled_01681.mp4
  official_distilled_01951.mp4
  ...
```

### 5.4 解码 GT

```bash
.venv/bin/python visual_multi_compare/decode_gt_from_precompute.py \
  --samples-json visual_multi_compare/sampled_8.json \
  --precompute-root /mnt/petrelfs/wanggongxuan/workspace/ode_precompute_stage2_8gpu_20260324T125612Z \
  --checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
  --output-dir /path/to/output_root/gt
```

输出目录形如：

```text
/path/to/output_root/gt/
  gt_01681.mp4
  gt_01951.mp4
  ...
```

## 6. 推荐执行顺序

建议按下面顺序跑：

1. 生成 `sampled_8.json`
2. 解码 GT
3. 跑 official distilled
4. 跑一个或多个 ODE checkpoint
5. 在同一批 8 个样本上做并排对比

## 7. 注意事项

- 这套流程的关键不是“统一一个 seed”，而是“每个样本使用它自己的 precompute stage-2 video seed”
- `sampled_8.json` 中已经把该 seed 写入 `seed` 字段，但脚本仍优先读取 `noise_seeds.stage2.video`，避免手工改坏 `seed`
- sbatch 文件名里仍有 `5p`，这是历史命名，不代表当前只跑 5 个样本
- `t208_5p_latest_step500_spot.sbatch` 的文件名是 `step500`，但当前配置指向的是 `step_01000` checkpoint；这是当前约定，不是 seed 对齐逻辑的一部分

## 8. 一句话总结

这次实现的是一套 **8 样本、按 precompute 的 `ode_noise_seeds.stage2.video` 做 seed 对齐** 的批量对比评估流程，包含样本生成、GT 解码、official distilled 推理和 ODE 推理四部分。
