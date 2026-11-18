"""Time-domain statistical features; no sampling rate or units are inferred."""
from __future__ import annotations

import math
import statistics
from typing import Sequence


def extract_statistical_features(samples: Sequence[Sequence[float]], channel_names: Sequence[str]) -> dict[str, float]:
    if len(samples) == 0 or len(channel_names) == 0:
        raise ValueError("samples and channel names must be non-empty")
    if any(len(row) != len(channel_names) for row in samples):
        raise ValueError("each sample row must match the configured channel count")
    result: dict[str, float] = {}
    for index, channel in enumerate(channel_names):
        values = [float(row[index]) for row in samples]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"channel {channel} contains a non-finite value")
        prefix = f"{channel}"
        result[f"{prefix}_mean"] = statistics.fmean(values)
        result[f"{prefix}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result[f"{prefix}_rms"] = math.sqrt(statistics.fmean(value * value for value in values))
        result[f"{prefix}_min"] = min(values)
        result[f"{prefix}_max"] = max(values)
        result[f"{prefix}_range"] = max(values) - min(values)
        result[f"{prefix}_median"] = statistics.median(values)
    return result
