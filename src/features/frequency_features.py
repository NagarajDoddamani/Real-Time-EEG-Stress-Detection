"""Welch band-power features, guarded by verified acquisition metadata."""
from __future__ import annotations

import math
from typing import Any, Sequence


def require_verified_sampling_frequency(config: dict[str, Any]) -> float:
    fs = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or fs is None:
        raise ValueError("frequency features require a verified sampling_frequency_hz")
    if not config.get("frequency_dependent_processing_permitted"):
        raise ValueError("frequency-dependent processing is disabled in configuration")
    fs = float(fs)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    return fs


def extract_welch_band_power(samples: Sequence[Sequence[float]], channel_names: Sequence[str],
                             config: dict[str, Any]) -> dict[str, float]:
    """Compute absolute Welch band powers using SciPy, if explicitly configured."""
    fs = require_verified_sampling_frequency(config)
    bands = config.get("frequency_bands_hz", {})
    if not bands:
        raise ValueError("frequency_bands_hz is empty")
    for name, bounds in bands.items():
        if len(bounds) != 2 or not 0 <= float(bounds[0]) < float(bounds[1]) < fs / 2:
            raise ValueError(f"band {name} must satisfy 0 <= low < high < Nyquist ({fs / 2:g} Hz)")
    try:
        from scipy.signal import welch
    except ImportError as exc:
        raise RuntimeError("SciPy is required for Welch PSD; install it only when frequency extraction is authorized") from exc
    if len(samples) < 2 or any(len(row) != len(channel_names) for row in samples):
        raise ValueError("samples must have at least two rows matching the channel configuration")
    result: dict[str, float] = {}
    for channel_index, channel in enumerate(channel_names):
        signal = [float(row[channel_index]) for row in samples]
        if not all(math.isfinite(value) for value in signal):
            raise ValueError(f"channel {channel} contains a non-finite value")
        frequencies, density = welch(signal, fs=fs, nperseg=min(len(signal), max(2, int(fs))))
        band_powers: dict[str, float] = {}
        ordered_bands = list(bands.items())
        for band_index, (name, (low, high)) in enumerate(ordered_bands):
            is_last_band = band_index == len(ordered_bands) - 1
            # Half-open edges avoid counting shared band boundaries twice.
            if is_last_band:
                selected = [(f, p) for f, p in zip(frequencies, density) if float(low) <= f <= float(high)]
            else:
                selected = [(f, p) for f, p in zip(frequencies, density) if float(low) <= f < float(high)]
            if len(selected) < 2:
                raise ValueError(f"window has insufficient Welch bins for band {name}")
            power = sum((selected[i + 1][0] - selected[i][0]) * (selected[i + 1][1] + selected[i][1]) / 2
                        for i in range(len(selected) - 1))
            band_powers[name] = power
        total = sum(band_powers.values())
        for name, power in band_powers.items():
            result[f"{channel}_{name}_power"] = power
            result[f"{channel}_{name}_relative_power"] = power / total if total > 0 else 0.0
    return result
