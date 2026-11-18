"""Deterministic participant-wise train/validation/test assignment."""
from __future__ import annotations

import random
from typing import Sequence


def split_participants(participant_ids: Sequence[str], seed: int = 42,
                       ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, list[str]]:
    unique = sorted(set(participant_ids))
    if len(unique) != len(participant_ids):
        raise ValueError("participant_ids contains duplicates")
    if len(unique) < 3:
        raise ValueError("at least 3 participants are required for disjoint train/validation/test sets")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be three non-negative values summing to 1")
    counts = [max(1, int(round(len(unique) * ratio))) for ratio in ratios]
    while sum(counts) > len(unique):
        index = max(range(3), key=lambda i: (counts[i] - 1, counts[i]))
        if counts[index] <= 1:
            raise ValueError("not enough participants for non-empty three-way split")
        counts[index] -= 1
    while sum(counts) < len(unique):
        index = max(range(3), key=lambda i: ratios[i] - counts[i] / len(unique))
        counts[index] += 1
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    a, b, c = counts
    result = {"train": sorted(shuffled[:a]), "validation": sorted(shuffled[a:a+b]), "test": sorted(shuffled[a+b:a+b+c])}
    if set(result["train"]) & set(result["validation"]) or set(result["train"]) & set(result["test"]) or set(result["validation"]) & set(result["test"]):
        raise RuntimeError("internal error: participant split overlaps")
    return result
