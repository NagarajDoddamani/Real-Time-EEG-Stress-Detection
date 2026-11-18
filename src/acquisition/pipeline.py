"""LSL metadata validation, rolling EEG windows, and trained-model inference."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.features.window_features import extract_window_features


@dataclass(frozen=True)
class StreamMetadata:
    name: str
    stream_type: str
    source_id: str
    channel_count: int
    nominal_srate: float
    channel_format: str
    channel_labels: list[str]
    stream_id: str = ""


def extract_stream_metadata(info: Any) -> StreamMetadata:
    """Extract metadata from a pylsl StreamInfo-like object, including XML labels."""
    labels: list[str] = []
    try:
        node = info.desc().child("channels").child("channel")
        for _ in range(int(info.channel_count())):
            labels.append((node.child_value("label") or "").strip())
            node = node.next_sibling()
    except (AttributeError, TypeError):
        labels = []
    fmt = info.channel_format()
    fmt_names = {0: "undefined", 1: "float32", 2: "double64", 3: "string",
                 4: "int32", 5: "int16", 6: "int8", 7: "int64"}
    if hasattr(fmt, "name"):
        fmt_name = str(fmt.name)
    else:
        try:
            fmt_name = fmt_names.get(int(fmt), str(fmt))
        except (TypeError, ValueError):
            fmt_name = str(fmt).removeprefix("cf_")
    return StreamMetadata(str(info.name()), str(info.type()), str(info.source_id()),
                          int(info.channel_count()), float(info.nominal_srate()),
                          str(fmt_name), labels, str(info.uid()))


def map_eeg_channels(labels: list[str], expected: list[str]) -> dict[str, int]:
    """Map exact channel labels to stream indices; reject missing/duplicate labels."""
    if not labels:
        raise ValueError("LSL stream has no channel labels; cannot safely map EEG channels")
    normalized = [label.strip().casefold() for label in labels]
    duplicates = sorted({v for v in normalized if normalized.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate LSL channel labels: {duplicates}")
    lookup = {label: index for index, label in enumerate(normalized) if label}
    missing = [name for name in expected if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"required EEG channels missing from LSL stream: {missing}")
    return {name: lookup[name.casefold()] for name in expected}


def estimate_rate(timestamps: Iterable[float], nominal_rate: float,
                  expected_rate: float, relative_tolerance: float = 0.02) -> dict[str, Any]:
    ts = np.asarray(list(timestamps), dtype=float)
    finite = ts[np.isfinite(ts)]
    intervals = np.diff(finite)
    positive = intervals[intervals > 0]
    observed = float(1.0 / np.median(positive)) if len(positive) else None
    median_interval = float(np.median(positive)) if len(positive) else None
    irregular = int(np.count_nonzero((intervals <= 0) | (intervals > 1.5 * median_interval))) if median_interval else 0
    compatible = (observed is not None and math.isclose(observed, expected_rate, rel_tol=relative_tolerance)
                 and (nominal_rate <= 0 or math.isclose(nominal_rate, expected_rate, rel_tol=relative_tolerance)))
    return {"nominal_rate_hz": float(nominal_rate), "observed_rate_hz": observed,
            "expected_rate_hz": float(expected_rate), "timestamp_count": int(len(ts)),
            "nonfinite_timestamp_count": int(np.count_nonzero(~np.isfinite(ts))),
            "nonpositive_interval_count": int(np.count_nonzero(intervals <= 0)),
            "irregular_or_gap_interval_count": irregular,
            "median_interval_seconds": median_interval, "compatible": bool(compatible)}


class RollingWindowBuffer:
    """Emit complete ordered windows at fixed sample steps, preserving endpoints."""
    def __init__(self, channel_count: int, window_samples: int = 2000, step_samples: int = 1000,
                 expected_rate_hz: float | None = None, rate_tolerance: float = 0.02):
        if min(channel_count, window_samples, step_samples) < 1 or step_samples > window_samples:
            raise ValueError("invalid rolling-buffer dimensions")
        self.channel_count, self.window_samples, self.step_samples = channel_count, window_samples, step_samples
        self.expected_rate_hz, self.rate_tolerance = expected_rate_hz, rate_tolerance
        self.samples: deque[np.ndarray] = deque()
        self.timestamps: deque[float] = deque()
        self.next_window_start = 0
        self.total_samples = 0
        self.buffer_start = 0

    def push(self, samples: Iterable[Iterable[float]], timestamps: Iterable[float]) -> list[tuple[np.ndarray, float, float, int]]:
        values = np.asarray(list(samples), dtype=np.float64)
        times = np.asarray(list(timestamps), dtype=np.float64)
        if values.size == 0:
            return []
        if values.ndim != 2 or values.shape[1] != self.channel_count or len(times) != len(values):
            raise ValueError("sample chunk shape or timestamp count does not match stream metadata")
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("samples and timestamps must be finite")
        out = []
        for row, stamp in zip(values, times):
            self.samples.append(row.copy()); self.timestamps.append(float(stamp)); self.total_samples += 1
            while self.next_window_start + self.window_samples <= self.total_samples:
                start_in_buffer = self.next_window_start - self.buffer_start
                rows = list(self.samples)
                stamps = list(self.timestamps)
                end = start_in_buffer + self.window_samples
                window_times = np.asarray(stamps[start_in_buffer:end], dtype=float)
                intervals = np.diff(window_times)
                if np.any(intervals <= 0):
                    raise ValueError("non-increasing LSL timestamps inside EEG window")
                median_interval = float(np.median(intervals))
                if np.any(intervals > 1.5 * median_interval):
                    raise ValueError("timestamp gap detected inside EEG window; inference stopped")
                if self.expected_rate_hz and not math.isclose(1.0 / median_interval,
                        self.expected_rate_hz, rel_tol=self.rate_tolerance):
                    raise ValueError("observed window rate is incompatible with trained-model rate")
                out.append((np.asarray(rows[start_in_buffer:end]), stamps[start_in_buffer], stamps[end - 1], self.next_window_start))
                self.next_window_start += self.step_samples
            # Drop only samples that cannot belong to the next window.
            while self.samples and self.buffer_start < self.next_window_start:
                self.samples.popleft(); self.timestamps.popleft(); self.buffer_start += 1
        return out


class TrainedModel:
    """Load the saved artifact and enforce its recorded feature/channel contract."""
    def __init__(self, model_path: Path, metadata_path: Path, config_path: Path):
        import joblib
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model = joblib.load(model_path)
        self.channels = list(self.metadata["channel_names_and_order"])
        self.feature_names = list(self.metadata["feature_names_and_order"])
        if self.channels != list(self.config["eeg_channels"]):
            raise ValueError("saved model channel order differs from EEG configuration")
        if int(self.metadata["sampling_frequency_hz"]) != int(self.config["sampling_frequency_hz"]):
            raise ValueError("saved model sampling rate differs from EEG configuration")
        if not hasattr(self.model, "n_features_in_") or self.model.n_features_in_ != len(self.feature_names):
            raise ValueError("model feature count differs from model metadata")
        if hasattr(self.model, "feature_names_in_") and list(self.model.feature_names_in_) != self.feature_names:
            raise ValueError("model feature names/order differ from model metadata")

    def predict_window(self, window: np.ndarray, start_timestamp: float, end_timestamp: float,
                       window_index: int) -> dict[str, Any]:
        features = self.extract_features(window)
        if list(features) != self.feature_names:
            raise ValueError("live feature names/order do not match saved model metadata")
        x = np.asarray([[features[key] for key in self.feature_names]], dtype=float)
        label = int(self.model.predict(x)[0])
        classes = [int(value) for value in getattr(self.model, "classes_", [])]
        result: dict[str, Any] = {"window_index": window_index, "window_start_timestamp": start_timestamp,
            "window_end_timestamp": end_timestamp, "prediction": label,
            "predicted_condition": {0: "baseline", 1: "stress"}.get(label, "unknown"),
            "prediction_text": {0: "Predicted baseline condition", 1: "Predicted stress condition"}.get(label, "Unknown class"),
            "model_algorithm": self.metadata.get("model_algorithm"),
            "model_version": self.metadata.get("model_created_utc"),
            "units_status": self.metadata.get("signal_units_status", "unverified")}
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)[0]
            result["baseline_probability"] = float(probabilities[classes.index(0)]) if 0 in classes else None
            result["stress_probability"] = float(probabilities[classes.index(1)]) if 1 in classes else None
        return result

    def extract_features(self, window: np.ndarray) -> dict[str, float]:
        """Use the same feature function and schema as offline training."""
        features = extract_window_features(window, self.channels, self.config)
        if list(features) != self.feature_names:
            raise ValueError("live feature names/order do not match saved model metadata")
        return features
