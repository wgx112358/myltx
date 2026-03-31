"""Shared seed resolution logic for batch inference scripts."""
from __future__ import annotations


def resolve_seed(sample: dict, fallback_seed: int) -> int:
    """Resolve seed with priority: noise_seeds.stage2.video > sample.seed > fallback."""
    ns = sample.get("noise_seeds")
    if ns is not None:
        try:
            return int(ns["stage2"]["video"])
        except (KeyError, TypeError):
            pass
    if "seed" in sample:
        return int(sample["seed"])
    return fallback_seed
