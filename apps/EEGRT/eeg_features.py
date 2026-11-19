"""Deterministic time and frequency features for one EEG window."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.signal import welch


def _welch_config(config: dict) -> tuple[int, int]:
    fs = float(config["sampling_frequency_hz"])
    psd = config.get("psd", {})
    segment_seconds = float(psd.get("segment_duration_seconds", 2.0))
    overlap_fraction = float(psd.get("overlap_fraction", 0.5))
    nperseg = int(round(fs * segment_seconds))
    if nperseg < 2 or not 0 <= overlap_fraction < 1:
        raise ValueError("invalid Welch segment duration or overlap")
    return nperseg, int(round(nperseg * overlap_fraction))


def feature_names(channel_names: Sequence[str], bands: dict[str, Sequence[float]]) -> list[str]:
    """Return the canonical feature order used for CSVs, CV and inference."""
    names: list[str] = []
    for channel in channel_names:
        names.extend([
            f"{channel}_mean", f"{channel}_std", f"{channel}_variance",
            f"{channel}_rms", f"{channel}_peak_to_peak",
        ])
        for band in bands:
            names.extend([f"{channel}_{band}_power", f"{channel}_{band}_relative_power"])
        names.append(f"{channel}_spectral_entropy")
    return names


def extract_window_features(samples: Sequence[Sequence[float]], channel_names: Sequence[str],
                            config: dict | float, bands: dict[str, Sequence[float]] | None = None) -> dict[str, float]:
    """Extract statistics, Welch band powers/relative powers and entropy.

    Welch uses the configured 2 s Hann segments with 50% overlap. Absolute
    features retain the input scale and must not be interpreted as calibrated
    physical measurements while EEG units remain unknown.
    """
    # The (rate, bands) call form remains supported for recorded replay.
    if not isinstance(config, dict):
        config = {
            "sampling_frequency_hz": float(config),
            "sampling_frequency_verified": True,
            "frequency_dependent_processing_permitted": True,
            "frequency_bands_hz": bands or {},
        }
    fs = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or fs is None:
        raise ValueError("frequency features require a verified sampling rate")
    fs = float(fs)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("sampling frequency must be finite and positive")
    if not config.get("frequency_dependent_processing_permitted"):
        raise ValueError("frequency-domain processing is disabled in configuration")
    if not channel_names:
        raise ValueError("channel_names must not be empty")

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != len(channel_names):
        raise ValueError("samples must be a 2D samples-by-channels window matching channel_names")
    if not np.isfinite(array).all():
        raise ValueError("window contains NaN or infinite EEG values")

    bands = config.get("frequency_bands_hz", {})
    if not bands:
        raise ValueError("frequency_bands_hz is empty")
    for name, bounds in bands.items():
        if len(bounds) != 2 or not 0 <= float(bounds[0]) < float(bounds[1]) < fs / 2:
            raise ValueError(f"{name} band must be below Nyquist ({fs / 2:g} Hz)")

    nperseg, noverlap = _welch_config(config)
    nperseg = min(nperseg, array.shape[0])
    noverlap = min(noverlap, nperseg - 1)
    result: dict[str, float] = {}
    for index, channel in enumerate(channel_names):
        signal = array[:, index]
        result[f"{channel}_mean"] = float(np.mean(signal))
        result[f"{channel}_std"] = float(np.std(signal, ddof=1)) if len(signal) > 1 else 0.0
        result[f"{channel}_variance"] = float(np.var(signal, ddof=1)) if len(signal) > 1 else 0.0
        result[f"{channel}_rms"] = float(np.sqrt(np.mean(np.square(signal))))
        result[f"{channel}_peak_to_peak"] = float(np.ptp(signal))

        frequencies, density = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            average="mean",
        )
        powers: dict[str, float] = {}
        ordered_bands = list(bands.items())
        for band_index, (band, bounds) in enumerate(ordered_bands):
            low, high = map(float, bounds)
            if band_index == len(ordered_bands) - 1:
                mask = (frequencies >= low) & (frequencies <= high)
            else:
                mask = (frequencies >= low) & (frequencies < high)
            if int(np.count_nonzero(mask)) < 2:
                raise ValueError(f"Welch PSD has fewer than two bins for {band}")
            powers[band] = float(np.trapezoid(density[mask], frequencies[mask]))

        total_band_power = sum(powers.values())
        for band, power in powers.items():
            result[f"{channel}_{band}_power"] = power
            result[f"{channel}_{band}_relative_power"] = power / total_band_power if total_band_power > 0 else 0.0

        entropy_mask = (frequencies >= 0.5) & (frequencies <= max(float(v[1]) for v in bands.values()))
        entropy_density = np.maximum(density[entropy_mask], 0.0)
        entropy_mass = float(entropy_density.sum())
        if entropy_mass <= 0 or len(entropy_density) <= 1:
            entropy = 0.0
        else:
            probabilities = entropy_density / entropy_mass
            probabilities = probabilities[probabilities > 0]
            entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(entropy_density)))
        result[f"{channel}_spectral_entropy"] = entropy

    expected = feature_names(channel_names, bands)
    if list(result) != expected:
        raise RuntimeError("internal feature ordering differs from the declared schema")
    return result
