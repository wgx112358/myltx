# Precompute Exact Replay

## 路径

文档路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/PRECOMPUTE_EXACT_REPLAY.md`

## 实现了什么

这套程序用于验证 precompute 保存的数据是否正确，核心思路是为同一批样本生成两份带声音的 mp4：

1. 从 precompute `.pt` 直接解码出的结果
2. 按 precompute 当时的生成配置重新跑一遍 distilled 生成流程得到的结果

这样可以直接比较：

- `.pt` 中保存的最终视频 latent / 音频 latent 是否能正确还原
- 用相同配置、相同 prompt、相同全局 index、相同 base seed 重放后，是否能得到与 `.pt` 一致的最终结果

## 涉及脚本

### 1. GT 解码脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/decode_gt_from_precompute.py`

当前行为：

- 同时读取：
  - `.precomputed/ode_video_trajectories/{idx}.pt`
  - `.precomputed/ode_audio_trajectories/{idx}.pt`
- 直接解码其中的 clean latent
- 输出带音频的 `gt_*.mp4`

也就是说，这个脚本现在不再只是“只看画面”，而是会把声音一起 mux 进 mp4。

### 2. Exact replay 脚本

路径：

`/mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/replay_precompute_exact.py`

当前行为：

- 读取 `samples.json`
- 读取 precompute 中同 index 的视频/音频 payload
- 先输出一份从 `.pt` 直接解码的 `pt_decode_*.mp4`
- 再加载 `ode/configs/gen_ode_data_distilled.yaml`
- 调用 `ode/gen_ode_data_distilled.py` 里的同一套生成逻辑重新生成
- 输出一份 `replay_*.mp4`

这份 replay 不是“近似推理”，而是直接复用 precompute 生成脚本里的：

- `generate_sample()`
- `encode_prompt()`
- 相同的 sigma schedule
- 相同的 config
- 相同的 `seed + global_idx -> per-stage-per-modality noise seeds` 推导逻辑

所以它比普通 `DistilledPipeline(seed=...)` 更适合做“数据是否保存正确”的校验。

## 为什么不能只用一个 `--seed`

precompute 原始生成脚本 `gen_ode_data_distilled.py` 不是简单地“所有阶段共用一个 generator”，而是按下面四组噪声分别生成：

- `stage1.video`
- `stage1.audio`
- `stage2.video`
- `stage2.audio`

这些 seed 由：

- `base seed`
- `global_idx`
- stage
- modality

共同导出。

所以如果只是普通 pipeline 里传一个 `--seed`，不一定能和 precompute 当时的内部噪声状态完全一致。`replay_precompute_exact.py` 的作用，就是避开这个误差源。

## 用法

### 1. 仅从 `.pt` 解码 GT

```bash
.venv/bin/python visual_multi_compare/decode_gt_from_precompute.py \
  --samples-json visual_multi_compare/sampled_8.json \
  --precompute-root /mnt/petrelfs/wanggongxuan/workspace/ode_precompute_stage2_8gpu_20260324T125612Z \
  --checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
  --output-dir /path/to/output/gt
```

输出：

```text
/path/to/output/gt/
  gt_01681.mp4
  gt_01951.mp4
  ...
```

这些 mp4 现在包含音频。

### 2. 同时生成 `.pt` 解码结果和 exact replay 结果

```bash
.venv/bin/python visual_multi_compare/replay_precompute_exact.py \
  --samples-json visual_multi_compare/sampled_8.json \
  --precompute-root /mnt/petrelfs/wanggongxuan/workspace/ode_precompute_stage2_8gpu_20260324T125612Z \
  --config ode/configs/gen_ode_data_distilled.yaml \
  --output-root /path/to/output/exact_replay \
  --device cuda
```

输出目录：

```text
/path/to/output/exact_replay/
  pt_decode/
    pt_decode_01681.mp4
    pt_decode_01951.mp4
    ...
  replay/
    replay_01681.mp4
    replay_01951.mp4
    ...
```

含义：

- `pt_decode/*.mp4`：从保存下来的 `.pt` 直接解码
- `replay/*.mp4`：按原始生成配置重新跑一遍得到

## 参数来源

`replay_precompute_exact.py` 的关键参数来自：

- 配置文件：
  - `ode/configs/gen_ode_data_distilled.yaml`
- precompute payload：
  - `.precomputed/ode_video_trajectories/*.pt`
  - `.precomputed/ode_audio_trajectories/*.pt`

其中配置文件负责提供：

- checkpoint
- gemma root
- spatial upsampler
- `stage1_height / stage1_width`
- `num_frames`
- `frame_rate`
- `seed`

而 sample 的 `index` 负责决定：

- prompt 对应哪条样本
- noise seed 派生使用哪个 `global_idx`

## 推荐验证方式

建议对每个样本做如下对比：

1. 听 `pt_decode_xxxxx.mp4`
2. 听 `replay_xxxxx.mp4`
3. 看画面是否一致
4. 听音频是否一致
5. 如果有差异，再进一步检查：
   - `.pt` 保存时是否已经偏了
   - decode 路径是否偏了
   - replay 配置是否不是当时那份

## 一句话总结

这套程序是为了做“precompute 数据正确性验证”而不是普通推理评估：它会输出两份带声音的结果，一份直接从 `.pt` 解码，一份按原始生成配置 exact replay，方便逐样本核对保存的数据是否可信。
