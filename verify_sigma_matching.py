"""
验证蒸馏模型 sigma 值与 LTX2Scheduler 不同步数的匹配关系。

用法:
    python verify_sigma_matching.py
    # 或自定义测试步数:
    python verify_sigma_matching.py --steps 30 40 50 100 160 200
"""
import argparse
import sys

sys.path.insert(0, "packages/ltx-core/src")
sys.path.insert(0, "packages/ltx-pipelines/src")

import torch
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES


def find_matching_indices(sigmas: torch.Tensor, target_sigmas: list[float]) -> list[tuple]:
    """对每个目标 sigma 在 sigmas 中找最近邻，返回 (target, actual, index, diff) 列表。"""
    results = []
    for t in target_sigmas:
        diffs = torch.abs(sigmas - t)
        idx = diffs.argmin().item()
        results.append((t, sigmas[idx].item(), idx, diffs[idx].item()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, nargs="+",
        default=[30, 40, 50, 100, 160, 200],
        help="要测试的步数列表"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LTX2Scheduler sigma 匹配验证")
    print("=" * 70)
    print(f"\n蒸馏模型 Stage1 sigma 值 ({len(DISTILLED_SIGMA_VALUES)} 个):")
    print(f"  {DISTILLED_SIGMA_VALUES}")
    print(f"\n蒸馏模型 Stage2 sigma 值 ({len(STAGE_2_DISTILLED_SIGMA_VALUES)} 个):")
    print(f"  {STAGE_2_DISTILLED_SIGMA_VALUES}")

    # ── 汇总表（max_diff 对比） ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("各步数 max_diff 汇总 (越小越好)")
    print(f"{'N':>6}  {'Stage1 max_diff':>16}  {'Stage2 max_diff':>16}  {'推荐':>6}")
    print("-" * 50)
    best_n, best_diff = None, float("inf")
    for N in args.steps:
        sigmas = LTX2Scheduler().execute(steps=N)
        m1 = max(r[3] for r in find_matching_indices(sigmas, DISTILLED_SIGMA_VALUES))
        m2 = max(r[3] for r in find_matching_indices(sigmas, STAGE_2_DISTILLED_SIGMA_VALUES))
        flag = "<-- best" if m1 < best_diff else ""
        if m1 < best_diff:
            best_diff = m1
            best_n = N
        print(f"{N:>6}  {m1:>16.6f}  {m2:>16.6f}  {flag}")

    # ── 最优 N 的详细匹配 ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"最优 N={best_n} 的完整 sigma 调度 ({best_n + 1} 个值):")
    sigmas_best = LTX2Scheduler().execute(steps=best_n)
    for i, s in enumerate(sigmas_best):
        print(f"  idx={i:3d}  sigma={s.item():.6f}")

    print(f"\nN={best_n} 下 Stage1 蒸馏 sigma 精确匹配:")
    print(f"  {'目标 sigma':>12}  {'调度中最近值':>14}  {'索引':>6}  {'误差':>10}")
    print("  " + "-" * 50)
    stage1_indices = []
    for target, actual, idx, diff in find_matching_indices(sigmas_best, DISTILLED_SIGMA_VALUES):
        stage1_indices.append(idx)
        print(f"  {target:>12.6f}  {actual:>14.6f}  {idx:>6}  {diff:>10.6f}")
    print(f"\n  Stage1 保存索引: {stage1_indices}")

    print(f"\nN={best_n} 下 Stage2 蒸馏 sigma 精确匹配:")
    print(f"  {'目标 sigma':>12}  {'调度中最近值':>14}  {'索引':>6}  {'误差':>10}")
    print("  " + "-" * 50)
    stage2_indices = []
    for target, actual, idx, diff in find_matching_indices(sigmas_best, STAGE_2_DISTILLED_SIGMA_VALUES):
        stage2_indices.append(idx)
        print(f"  {target:>12.6f}  {actual:>14.6f}  {idx:>6}  {diff:>10.6f}")
    print(f"\n  Stage2 保存索引 (相对全局): {stage2_indices}")
    print(f"  Stage2 起始索引 (教师从此开始运行): {stage2_indices[0]}")


if __name__ == "__main__":
    main()
