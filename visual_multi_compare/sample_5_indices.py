#!/usr/bin/env python3
# ruff: noqa: T201
"""Sample indices from the prompt CSV and extract per-sample noise seeds from precomputed data.

Produces a ``sampled_8.json`` file used by the batch inference scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = REPO_ROOT / "datagen" / "ltx_prompts_v1_12000.csv"
DEFAULT_PRECOMPUTE_ROOT = Path(
    "/mnt/petrelfs/wanggongxuan/workspace/ode_precompute_stage2_8gpu_20260324T125612Z"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sampled_8.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample indices and extract noise seeds")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--precompute-root", type=Path, default=DEFAULT_PRECOMPUTE_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--total", type=int, default=12000)
    return parser.parse_args(argv)


def load_prompts(csv_path: Path) -> list[str]:
    """Load prompts from the CSV (single-column ``text_prompt``)."""
    prompts: list[str] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            prompts.append(row["text_prompt"])
    return prompts


def extract_noise_seeds(pt_path: Path) -> dict | None:
    """Extract ``ode_noise_seeds`` dict from a precomputed ``.pt`` file if present."""
    if not pt_path.exists():
        return None
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    return payload.get("ode_noise_seeds")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    prompts = load_prompts(args.csv_path)
    assert len(prompts) >= args.total, f"CSV has {len(prompts)} rows, expected >= {args.total}"

    rng = random.Random(args.sample_seed)
    indices = sorted(rng.sample(range(args.total), args.num_samples))

    traj_dir = args.precompute_root / ".precomputed" / "ode_video_trajectories"

    samples: list[dict] = []
    for idx in indices:
        prompt = prompts[idx]
        noise_seeds = extract_noise_seeds(traj_dir / f"{idx:05d}.pt")

        # Derive the stage-2 video seed for the top-level ``seed`` field
        seed: int | None = None
        if noise_seeds is not None:
            try:
                seed = int(noise_seeds["stage2"]["video"])
            except (KeyError, TypeError):
                pass

        entry: dict = {"index": idx, "prompt": prompt}
        if seed is not None:
            entry["seed"] = seed
        if noise_seeds is not None:
            entry["noise_seeds"] = noise_seeds
        samples.append(entry)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples to {args.output_path}")
    for s in samples:
        print(f"  idx={s['index']:05d}  seed={s.get('seed', 'N/A')}  prompt={s['prompt'][:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
