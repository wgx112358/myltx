# LTX-2.3 Distilled ODE 轨迹数据说明



## 1. 数据概览

该数据由蒸馏版 LTX-2.3 两阶段推理流程生成，保存的是 **ODE 去噪过程中的 latent 轨迹**，不是直接解码后的像素视频或波形音频。

当前默认配置见 [`gen_ode_data_distilled.yaml`](/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/configs/gen_ode_data_distilled.yaml)，对应特征如下：

- 模型：`ltx-2.3-22b-distilled.safetensors`
- 文本输入：[`myltx/datagen/ltx_prompts_12000.csv`](/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/datagen/ltx_prompts_12000.csv) 的 `text_prompt` 列
- Stage 1 分辨率：`512 x 768`
- Stage 2 分辨率：`1024 x 1536`
- 帧数：`121`
- 帧率：`24.0`
- 随机种子基准：`42`

基于当前本地已生成结果，数据集包含：

- 样本数：`12000`
- 文件命名范围：`00000.pt` 到 `11999.pt`
- 本地目录：[`myltx/ode/data_distilled`](/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled)
- 当前目录体积：约 `446G`

在线版本已上传到 ModelScope：

- 数据集地址：<https://www.modelscope.cn/datasets/logan112/ltx2-3-distilled>

## 2. 生成逻辑

### Stage 1

Stage 1 在半分辨率 `512 x 768` 上运行蒸馏版固定 sigma 调度，共 8 步去噪，保存 9 个状态：

```text
[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
```

这 9 个状态包含：

- 第 0 个状态：初始噪声 latent
- 第 1 到第 8 个状态：每一步 Euler 更新后的 latent

### Stage 2

Stage 2 使用 Stage 1 最终 clean latent 的上采样结果作为起点，再加噪到 `sigma = 0.909375`，然后执行 3 步去噪，保存 4 个状态：

```text
[0.909375, 0.725, 0.421875, 0.0]
```

蒸馏模型版本有两个重要特点：

- 不使用 CFG
- 不需要负向 prompt

## 3. 单个 `.pt` 文件格式

每个样本保存为一个独立的 PyTorch 文件，命名格式为：

```text
{index:05d}.pt
```

例如：

```text
00000.pt
00001.pt
...
11999.pt
```

单个样本加载后是一个 `dict`，当前已验证的键如下：

| 键名 | 类型 | shape / 内容 | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `prompt` | `str` | 文本 prompt | - | 当前样本对应的输入文本 |
| `stage1_video_traj` | `Tensor` | `[9, 128, 16, 16, 24]` | `torch.bfloat16` | Stage 1 视频 latent 轨迹 |
| `stage1_audio_traj` | `Tensor` | `[9, 8, 126, 16]` | `torch.bfloat16` | Stage 1 音频 latent 轨迹 |
| `stage1_sigmas` | `Tensor` | `[9]` | `torch.float32` | Stage 1 sigma 调度 |
| `stage2_video_traj` | `Tensor` | `[4, 128, 16, 32, 48]` | `torch.bfloat16` | Stage 2 视频 latent 轨迹 |
| `stage2_audio_traj` | `Tensor` | `[4, 8, 126, 16]` | `torch.bfloat16` | Stage 2 音频 latent 轨迹 |
| `stage2_sigmas` | `Tensor` | `[4]` | `torch.float32` | Stage 2 sigma 调度 |

其中：

- `stage1_video_traj[0]` 和 `stage2_video_traj[0]` 都表示对应阶段的起始 noisy latent
- `stage1_video_traj[-1]` 和 `stage2_video_traj[-1]` 都表示对应阶段最终去噪后的 latent
- 音频轨迹与视频轨迹按同一步数对齐
- 所有轨迹都保存在 **latent 空间**，不是像素空间或原始波形空间

## 4. 目录结构

默认配置下，主要会出现两个目录。

### 4.1 最终数据目录

```text
myltx/ode/data_distilled/
├── 00000.pt
├── 00001.pt
├── ...
└── 11999.pt
```

这里的 `.pt` 文件就是最终训练或分析时需要使用的数据。

### 4.2 Prompt context 缓存目录

```text
myltx/ode/data_distilled_prompt_ctx_cache/
└── chunk_0000/
    ├── 00000.pt
    ├── 00001.pt
    └── ...
```

这个目录保存的是 prompt 预编码缓存，不属于最终 ODE 数据样本。缓存文件内部通常包含：

- `video_encoding`
- `audio_encoding`
- `attention_mask`

在当前配置里，`delete_prompt_ctx_after_use: true`，因此样本成功写入后，对应缓存一般会被删除，避免磁盘占用持续增长。


## 5. 读取示例

```python
import torch

sample = torch.load("00000.pt", map_location="cpu")

print(sample.keys())
print(sample["prompt"])
print(sample["stage1_video_traj"].shape, sample["stage1_video_traj"].dtype)
print(sample["stage1_audio_traj"].shape, sample["stage1_audio_traj"].dtype)
print(sample["stage1_sigmas"])
print(sample["stage2_video_traj"].shape, sample["stage2_video_traj"].dtype)
print(sample["stage2_audio_traj"].shape, sample["stage2_audio_traj"].dtype)
print(sample["stage2_sigmas"])
```

## 6. 使用注意事项

- 这些数据是 ODE 轨迹 latent，不是可直接播放的视频文件。
- 若下游任务需要可视化，需要结合 LTX-2.3 的解码流程把 latent 解码回视频或音频。
- shape 与当前默认配置强相关；如果修改分辨率、帧数或生成脚本，后续新数据的 shape 可能变化。
- 当前样本中的 tensor 主要使用 `bfloat16`，读取和后处理时需要注意 dtype。

## 7. 一句话总结

`logan112/ltx2-3-distilled` 是一个基于 LTX-2.3 distilled 两阶段流程生成的 **音视频联合 ODE latent 轨迹数据集**；每条样本是一个 `.pt` 字典，包含文本 prompt、Stage 1/Stage 2 的视频轨迹、音频轨迹以及对应的 sigma 序列。
