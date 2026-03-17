# 当前 ODE Regression 训练代码说明

## 1. 当前状态结论

当前这套 ODE regression 训练代码，在我已经完成的验证范围内，没有发现阻断训练的逻辑错误。

这里的“验证范围”不是只看静态代码，而是实际跑过了下面这条闭环：

1. 使用 `gen_ode_data_distilled.py` 真实生成 ODE `.pt` 样本
2. 使用 `convert_ode_pt_to_precomputed.py` 转成 `.precomputed`
3. 使用 `ltx-trainer` 的 `ode_regression` 策略跑 1 个真实训练 step

同时还补了这些额外检查：

- block-causal self-attn / AV cross-attn mask 是否真正进入模型
- audio / video `ode_sigma` 是否强校验一致
- `ode_noise_seeds` 是否在 audio / video 间一致
- 故意篡改 `ode_noise_seeds` 后 trainer 是否会拒绝脏数据

需要诚实说明的是：

- 我不能保证“绝对没有任何 bug”
- 但就当前已经跑通的真实链路来看，训练主路径是通的，逻辑也和当前设计目标一致

## 2. 整体流程

当前 ODE regression 的完整链路是：

1. 生成 ODE 轨迹数据
2. 转换为 `ltx-trainer` 可读取的 `.precomputed`
3. 使用 `ode_regression` 训练策略训练 transformer

对应文件：

- 数据生成
  - `myltx/ode/gen_ode_data.py`
  - `myltx/ode/gen_ode_data_distilled.py`
- 数据转换
  - `myltx/ode/convert_ode_pt_to_precomputed.py`
- 训练策略
  - `myltx/packages/ltx-trainer/src/ltx_trainer/training_strategies/ode_regression.py`
- attention plumbing
  - `myltx/packages/ltx-core/src/ltx_core/model/transformer/modality.py`
  - `myltx/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
  - `myltx/packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
  - `myltx/packages/ltx-core/src/ltx_core/model/transformer/attention.py`

## 3. 当前训练目标

训练输入和监督目标仍然保持在 LTX 原生的 velocity 预测空间里：

- 输入：当前轨迹状态 `x_t`
- 干净目标：`x0`
- 噪声强度：`sigma`
- 训练目标：`(x_t - x0) / sigma`

也就是说，trainer 不重新设计模型 head，而是把 clean latent 转成 velocity target 后继续训练现有 transformer。

## 4. 当前数据格式

### 4.1 原始 ODE `.pt` 样本

当前生成脚本输出的原始样本，核心字段包括：

- `prompt`
- `stage1_video_traj`
- `stage1_audio_traj`
- `stage1_sigmas`
- `stage2_video_traj`
- `stage2_audio_traj`
- `stage2_sigmas`
- `noise_seeds`

其中 `noise_seeds` 是当前新增的噪声元信息，用于明确记录：

- base seed
- sample 的 `global_idx`
- `stage1.video`
- `stage1.audio`
- `stage2.video`
- `stage2.audio`

### 4.2 `.precomputed` 样本

转换后，目录结构为：

```text
<dataset_root>/
├── .precomputed/
│   ├── latents/
│   ├── audio_latents/
│   └── conditions/
└── conversion_manifest.json
```

在 `ode_regression` 模式下：

- `latents/*.pt` 会写入
  - `latents`
  - `ode_target_latents`
  - `ode_sigma`
  - `ode_step_index`
  - `ode_clean_step_index`
  - `ode_noise_seeds`
- `audio_latents/*.pt` 也会写入同样的 ODE 监督字段

## 5. 训练策略当前实现

### 5.1 入口

训练策略入口是：

- `myltx/packages/ltx-trainer/src/ltx_trainer/training_strategies/ode_regression.py`

这个策略做的事情是：

1. 读取当前 video/audio latent
2. 读取 clean target latent
3. 读取 `ode_sigma`
4. 生成 per-token timestep
5. 构造 block-causal masks
6. 组织成 `Modality`
7. 计算 video/audio 联合损失

### 5.2 配置项

当前默认启用的核心配置包括：

- `use_block_causal_mask=True`
- `block_size=6`
- `independent_first_frame=True`
- `audio_boundary_mode="left"`
- `local_attn_size=-1`
- `validate_audio_sigma_match=True`
- `sigma_match_atol=1e-6`
- `sigma_match_rtol=1e-5`

含义如下：

- 首帧单独成块
- 后续每 `6` 个 latent frame 为一个块
- audio token 默认按 `left boundary` 对齐到 video block
- self/cross attn 默认不做局部窗口裁剪
- audio/video sigma 必须一致

## 6. 当前 block-causal 逻辑

### 6.1 video self-attn

规则是：

- 同块内双向可见
- 只能看本块和历史块
- 不能看未来块

### 6.2 audio self-attn

规则和 video 一致：

- 同块内双向可见
- 跨块严格因果

### 6.3 AV cross-attn

当前已经不是无约束 cross-attn，而是 block-causal cross-attn：

- `video -> audio`
  - video query 只能看到本块及历史块的 audio key
- `audio -> video`
  - audio query 只能看到本块及历史块的 video key

### 6.4 text cross-attn

文本 cross-attn 当前没有加 temporal causal mask。

这是刻意保留的当前设计：

- text 只继续使用 prompt padding mask
- temporal causal 只作用于 video/audio token 本身，以及 AV cross-attn

## 7. 当前 position encoding 语义

当前没有硬套论文里的固定 `31/157` 比例。

当前实现保持 `myltx` 自己的时间语义：

- video 继续使用当前秒级时间坐标
- audio 继续使用 patchifier 生成的秒级时间坐标
- AV cross-attn 的 temporal RoPE 继续只看时间轴

为什么这样做：

- 你当前真实 audio latent 是 `126 tokens / 5.04s`
- 如果强行套论文常数，会和当前 VAE / patchifier 的时间定义冲突

## 8. 当前 sigma 与损失逻辑

### 8.1 sigma 来源

当前 trainer 不在线采样 sigma。

它只读取预计算数据里的：

- `latents["ode_sigma"]`
- `audio_latents["ode_sigma"]`

### 8.2 audio sigma

之前的一个风险点是 audio 侧可能偷偷复用 video sigma。

当前已经改成：

- audio 单独读取自己的 `ode_sigma`
- 用 `torch.allclose` 和 video sigma 校验
- 如果不一致，直接 `ValueError`

### 8.3 loss mask

当前 loss mask 基于各自 sigma 是否大于 `sigma_epsilon`：

- video loss mask 用 video sigma
- audio loss mask 用 audio sigma

所以：

- `sigma <= epsilon` 的样本不会参与对应损失

## 9. 当前噪声逻辑

### 9.1 “不同 chunk 之间噪声是否独立”

这个问题要分两层理解。

#### 第一层：时序块内部的 token 噪声

底层 `GaussianNoiser` 仍然是对整个 latent tensor 做一次 `torch.randn(...)`。

这意味着：

- 每个元素本身就是独立高斯噪声
- 不存在“同一个 temporal block 共用一份噪声”的实现
- 所以从统计意义上说，block 与 block 之间本来就是独立采样的

#### 第二层：stage / modality 级别的 RNG 设计

这一层是本次重点修的地方。

之前的问题是：

- Stage1 和 Stage2 更像是在复用同一条 RNG 流
- 这会让 Stage2 起始噪声受 Stage1 内部随机数消耗顺序影响

当前已经改成显式派生 seed：

- `stage1.video`
- `stage1.audio`
- `stage2.video`
- `stage2.audio`

也就是说，现在是：

- stage 独立
- modality 独立
- seed 可追踪
- 生成逻辑变化不会偷偷影响另一个 stage 的起始噪声

### 9.2 `noise_seeds` / `ode_noise_seeds`

当前链路中：

- 原始生成样本写 `noise_seeds`
- converter 透传为 `ode_noise_seeds`
- trainer 首个 batch 会打印 debug
- trainer 会检查 audio/video `ode_noise_seeds` 是否一致

如果 audio/video 的噪声元信息不一致，trainer 会直接报错，不会继续训练。

## 10. attention backend 当前实现

当前 `BlockMask` 不再只是构造出来摆设，而是真的进入 backend。

逻辑是：

- dense mask 继续走原有 additive bias 路径
- `BlockMask` 透传，不 dense 化
- 如果收到 `BlockMask`
  - 强制走 PyTorch `flex_attention`
  - 不让默认 backend 或 `xformers` 抢走

这样做的目的：

- 避免构造 `[B, T, T]` 巨型 dense mask
- 让长序列 block-causal 训练真正可跑

## 11. 当前已经做过的真实验证

### 11.1 训练策略级验证

已经确认：

- `video.attention_mask` 是 `BlockMask`
- `audio.attention_mask` 是 `BlockMask`
- `video.cross_attention_mask` 是 `BlockMask`
- `audio.cross_attention_mask` 是 `BlockMask`

### 11.2 噪声元信息验证

已经确认：

- 原始 `gen_ode_data_distilled.py` 真实生成的 `.pt` 样本包含 `noise_seeds`
- 转换后的 `.precomputed` video/audio payload 都包含 `ode_noise_seeds`
- trainer 能打印首个样本的 noise metadata
- 故意篡改 audio `ode_noise_seeds` 后，trainer 会报错

### 11.3 端到端闭环验证

已经真实跑通：

1. `gen_ode_data_distilled.py` 生成 1 条 `.pt`
2. `convert_ode_pt_to_precomputed.py` 转出 3 条 `stage2/all_non_last` 训练样本
3. `ltx-trainer` 跑 1 个 step
4. 成功保存 LoRA 权重

这说明：

- 生成
- 转换
- 数据加载
- mask 构造
- forward
- backward
- checkpoint 保存

这一整条链路是通的

## 12. 当前还保留的边界与假设

下面这些不是 bug，但属于当前实现边界：

- 当前真实端到端闭环验证跑的是 distilled 生成脚本
- non-distilled `gen_ode_data.py` 我已经改了同样的噪声 seed 设计，但没有再跑完整长任务
- 当前没有做长时间多 step 训练稳定性统计，只做了真实 1-step 闭环
- 文本 cross-attn 仍不做 temporal causal
- `local_attn_size > 0` 的窗口裁剪逻辑已实现，但当前主验证使用的是默认 `-1`

## 13. 训练时如何看 loss

当前训练时的 loss 展示分成两路：

- 终端进度条
  - 默认就会显示当前训练 loss
- Weights & Biases
  - 会记录 `train/loss`
  - 可选记录 `train/loss_ema`
  - 同时记录 `train/learning_rate`
  - 同时记录 `train/step_time`

### 13.1 当前新增的 W&B 配置

`packages/ltx-trainer/configs/ltx2_av_ode_regression.yaml` 里现在支持：

- `wandb.enabled`
  - 是否启用 W&B
- `wandb.project`
  - W&B project 名称
- `wandb.entity`
  - 你的用户名或 team
- `wandb.name`
  - run 名称
  - 为空时默认使用 `output_dir` 的目录名
- `wandb.mode`
  - `online` 或 `offline`
  - `offline` 适合先本地训练，之后再 `wandb sync`
- `wandb.train_log_interval`
  - 每多少个 optimizer step 向 W&B 记录一次训练指标
- `wandb.console_log_interval`
  - 关闭进度条时，每多少个 optimizer step 在终端打印一次 loss
- `wandb.log_loss_ema`
  - 是否额外记录平滑 loss
- `wandb.loss_ema_beta`
  - loss EMA 的平滑系数

### 13.2 当前 loss 记录方式

当前不是简单记录“最后一个 micro-batch 的瞬时 loss”，而是：

- 先在一个 gradient accumulation 周期内累计 loss
- 到 optimizer step 时取平均
- 再把这个 optimizer-step loss 记录到 W&B

这样做的好处是：

- 当 `gradient_accumulation_steps > 1` 时，loss 曲线更符合真实训练步
- W&B 的 step 轴会对齐 `train/global_step`

### 13.3 当前建议

如果你主要想看训练走势，建议：

- `wandb.train_log_interval=1`
- `wandb.log_loss_ema=true`
- `wandb.loss_ema_beta=0.98`

如果你希望更高频地在纯终端里看到 loss，可以：

- 启动训练时加 `--disable-progress-bars`
- 同时把 `wandb.console_log_interval` 调小，比如 `1` 或 `5`

## 14. 推荐使用方式

如果你要训练当前版本的 ODE regression，建议：

1. 数据导出时使用 `--export-mode ode_regression`
2. 训练样本展开方式使用 `--trajectory-step all_non_last`
3. 训练阶段保持 `use_block_causal_mask=True`
4. 保持 `validate_audio_sigma_match=True`
5. 不要手动删除 `ode_noise_seeds`
6. 需要在线看 loss 时，保持 `wandb.enabled=True`

## 15. 一句话总结

当前这套 ODE regression 训练代码，已经从“能跑”推进到“训练逻辑、mask、sigma、噪声元信息都对齐，而且真实闭环跑通过”。

如果后续还要继续增强，我认为最自然的方向只有两个：

- 补更系统的单元测试
- 做更长训练和更大数据规模下的稳定性验证
