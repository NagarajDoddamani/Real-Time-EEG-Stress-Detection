"""Combine feature dictionaries with deterministic names and order."""
from __future__ import annotations

from typing import Mapping


def aggregate_feature_groups(*groups: Mapping[str, float]) -> tuple[list[str], list[float]]:
    merged: dict[str, float] = {}
    for group in groups:
        for name, value in group.items():
            if name in merged:
                raise ValueError(f"duplicate feature name: {name}")
            merged[name] = float(value)
    names = sorted(merged)
    return names, [merged[name] for name in names]
