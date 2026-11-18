#!/usr/bin/env python3
"""Replay normalized EEG in ordered chunks through the streaming window path.

The replay uses the same Welch function as offline feature construction. A
candidate sample rate is allowed for experiments but remains labelled as a
hypothesis. Without a model, it writes feature windows only; with a compatible
model, it also writes predictions. This does not connect to Mitsar hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURE_SOURCE = ROOT / "src" / "features"
if str(FEATURE_SOURCE) not in sys.path:
    sys.path.insert(0, str(FEATURE_SOURCE))
from window_features import extract_window_features, feature_names as canonical_feature_names


def load_normalized(path: Path, configured: list[str], selected: list[str]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if header != configured:
        raise ValueError("input must be a normalized CSV with the configured channel header and order")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(configured) or not len(data):
        raise ValueError("input must contain samples × configured EEG channels")
    if not np.isfinite(data).all():
        raise ValueError("input contains NaN or infinite EEG values")
    return data[:, [configured.index(channel) for channel in selected]]


def choose_rate(config: dict[str, Any], hypothesis: float | None) -> tuple[float, str]:
    if hypothesis is not None:
        rate = float(hypothesis)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sampling-rate hypothesis must be finite and positive")
        return rate, "hypothesis"
    rate = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or rate is None:
        raise ValueError("sampling rate is unknown; pass an explicit --sampling-rate-hypothesis for replay experiments")
    rate = float(rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("verified sampling rate must be finite and positive")
    return rate, "verified"


def feature_row(signal_window: np.ndarray, channels: list[str], config: dict[str, Any],
                fs: float, start: int, labeler: Any = None,
                feature_names: list[str] | None = None) -> dict[str, Any]:
    features = extract_window_features(signal_window, channels, fs, config["frequency_bands_hz"])
    row: dict[str, Any] = {
        "window_start_sample": start,
        "window_end_sample_exclusive": start + len(signal_window),
        "window_start_seconds_at_replay_rate": start / fs,
        **features,
    }
    if labeler is not None:
        ordered_names = feature_names or list(features)
        model_input = [[features[name] for name in ordered_names]]
        predicted = int(labeler.predict(model_input)[0])
        row["predicted_label"] = predicted
        row["predicted_condition"] = {int(value): key for key, value in config["label_map"].items()}.get(predicted, "unknown")
        if hasattr(labeler, "predict_proba"):
            probabilities = labeler.predict_proba(model_input)[0]
            classes = [int(value) for value in labeler.classes_]
            row["probability_baseline"] = float(probabilities[classes.index(0)]) if 0 in classes else 0.0
            row["probability_stress"] = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    return row


def replay(data: np.ndarray, channels: list[str], config: dict[str, Any], fs: float,
           chunk_samples: int, replay_speed: float, fast: bool,
           labeler: Any = None, feature_names: list[str] | None = None,
           start_sample_index: int = 0) -> list[dict[str, Any]]:
    if chunk_samples < 1 or replay_speed <= 0:
        raise ValueError("chunk_samples and replay_speed must be positive")
    window_len = int(round(fs * float(config["window_duration_seconds"])))
    step = max(1, int(round(window_len * (1 - float(config["window_overlap_fraction"])))))
    if window_len < 2:
        raise ValueError("configured window length resolves to fewer than two samples")
    buffer = np.empty((0, len(channels)), dtype=np.float64)
    if start_sample_index < 0:
        raise ValueError("start_sample_index cannot be negative")
    buffer_start = int(start_sample_index)
    consumed = 0
    rows: list[dict[str, Any]] = []
    replay_start = time.perf_counter()

    for chunk_start in range(0, len(data), chunk_samples):
        chunk = data[chunk_start:chunk_start + chunk_samples]
        buffer = np.concatenate((buffer, chunk), axis=0)
        consumed += len(chunk)
        while len(buffer) >= window_len:
            window = buffer[:window_len]
            row = feature_row(window, channels, config, fs, buffer_start, labeler, feature_names)
            rows.append(row)
            buffer = buffer[step:]
            buffer_start += step
        if not fast:
            deadline = replay_start + consumed / (fs * replay_speed)
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="normalized EEG CSV in original sample order")
    parser.add_argument("--sampling-rate-hypothesis", type=float,
                        help="explicit exploratory sample rate; does not verify device metadata")
    parser.add_argument("--chunk-samples", type=int, default=64)
    parser.add_argument("--replay-speed", type=float, default=1.0, help="1.0 paces to the selected sample-rate value")
    parser.add_argument("--duration-seconds", type=float,
                        help="optional maximum replay duration; the source sample order is unchanged")
    parser.add_argument("--start-sample", type=int, default=0,
                        help="zero-based source sample index to begin replay from")
    parser.add_argument("--fast", action="store_true", help="process immediately without wall-clock pacing")
    parser.add_argument("--model", type=Path, help="optional joblib classifier for prediction replay")
    parser.add_argument("--training-metadata", type=Path, help="metadata JSON next to the selected model")
    parser.add_argument("--allow-exploratory-model", action="store_true",
                        help="permit replay with a model marked not deployment-ready; results remain exploratory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eeg_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "replay_features.csv")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        channels = list(config["eeg_channels"])
        if len(channels) != 19 or len(set(channels)) != 19:
            raise ValueError("configuration must define the 19 unique training EEG channels")
        fs, rate_status = choose_rate(config, args.sampling_rate_hypothesis)
        for band, bounds in config["frequency_bands_hz"].items():
            if not 0 <= float(bounds[0]) < float(bounds[1]) < fs / 2:
                raise ValueError(f"{band} band exceeds the selected Nyquist frequency")
        data = load_normalized(args.input, config["eeg_channels"], channels)
        source_sample_count = len(data)
        if args.start_sample < 0 or args.start_sample >= source_sample_count:
            raise ValueError(f"--start-sample must be between 0 and {source_sample_count - 1}")
        data = data[args.start_sample:]
        if args.duration_seconds is not None:
            if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
                raise ValueError("--duration-seconds must be finite and positive")
            data = data[:min(len(data), int(round(args.duration_seconds * fs)))]

        labeler = None
        training_metadata = {}
        if args.model:
            if not args.training_metadata:
                raise ValueError("--training-metadata is required with --model")
            training_metadata = json.loads(args.training_metadata.read_text(encoding="utf-8"))
            if not training_metadata.get("deployment_ready", True) and not args.allow_exploratory_model:
                raise ValueError("model is marked exploratory; pass --allow-exploratory-model for replay experiments")
            metadata_channels = training_metadata.get("channel_names_and_order", training_metadata.get("channel_order"))
            if metadata_channels != channels:
                raise ValueError("model channel order does not match replay channel order")
            feature_names = training_metadata.get("feature_names_and_order", training_metadata.get("feature_names"))
            if not feature_names:
                raise ValueError("training metadata lacks feature_names")
            expected_features = canonical_feature_names(channels, config["frequency_bands_hz"])
            if feature_names != expected_features:
                raise ValueError("model feature names/order do not match the current channel and band configuration")
            trained_rate = training_metadata.get("sampling_frequency_hz")
            if trained_rate is not None and not math.isclose(float(trained_rate), fs):
                raise ValueError("model and replay sample-rate values differ")
            import joblib
            labeler = joblib.load(args.model)
        else:
            feature_names = None

        rows = replay(data, channels, config, fs, args.chunk_samples, args.replay_speed,
                      args.fast, labeler, feature_names, args.start_sample)
        if not rows:
            raise ValueError("recording did not contain a complete configured window")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "input": str(args.input), "output": str(args.output), "channel_order": channels,
            "sampling_rate_hz": fs, "sampling_rate_status": rate_status,
            "chunk_samples": args.chunk_samples, "replay_speed": args.replay_speed,
            "source_sample_count": source_sample_count,
            "processed_sample_count": len(data),
            "source_start_sample_index": args.start_sample,
            "source_end_sample_index_exclusive": args.start_sample + len(data),
            "paced_duration_seconds": len(data) / (fs * args.replay_speed),
            "window_count": len(rows), "prediction_enabled": labeler is not None,
            "prediction_status": "exploratory; model metadata is not deployment-ready" if labeler is not None and not training_metadata.get("deployment_ready", True) else "model-backed" if labeler is not None else "features only",
            "model_training_metadata": str(args.training_metadata) if args.model else None,
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Replayed {len(data)} of {source_sample_count} samples from source index {args.start_sample}; "
              f"emitted {len(rows)} windows "
              f"({fs:g} Hz {rate_status} rate; speed x{args.replay_speed:g}).")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Replay blocked: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
