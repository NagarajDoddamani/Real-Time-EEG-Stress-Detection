"""Read EEG sessions from Neo4j and predict with the saved training pipeline.

This module is the local Python service used by the Device 2 Electron UI. It
never acquires EEG or changes samples; it reads stored sample chunks and calls
the exact feature and model code used by the existing inference pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "eeg_config.json"
MODEL_PATH = ROOT / "models" / "stress_model.pkl"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"


def _components():
    from src.acquisition.pipeline import RollingWindowBuffer, TrainedModel, map_eeg_channels
    from src.storage.neo4j_store import Neo4jReader
    return RollingWindowBuffer, TrainedModel, map_eeg_channels, Neo4jReader


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return sessions that have raw EEG chunks available for Device 2."""
    _, _, _, Reader = _components()
    reader = Reader()
    try:
        rows = reader.list_sessions(limit)
    finally:
        reader.close()
    result = []
    for row in rows:
        session = row["session"]
        if int(row.get("raw_sample_count", 0)) <= 0:
            continue
        result.append({
            "session_id": session.get("session_id"),
            "started_at_utc": session.get("started_at_utc"),
            "device": session.get("device", "unknown"),
            "channel_count": session.get("channel_count"),
            "nominal_sampling_rate_hz": session.get("nominal_sampling_rate_hz"),
            "observed_sampling_rate_hz": session.get("observed_sampling_rate_hz"),
            "raw_sample_count": int(row.get("raw_sample_count", 0)),
            "raw_chunk_count": int(row.get("raw_chunk_count", 0)),
            "prediction_count_already_stored": int(row.get("prediction_count", 0)),
        })
    return result


def predict_session(session_id: str) -> dict[str, Any]:
    """Retrieve one stored recording and classify complete 4 s / 50% windows."""
    RollingWindowBuffer, TrainedModel, map_eeg_channels, Reader = _components()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fs = float(config["sampling_frequency_hz"])
    if not config.get("sampling_frequency_verified"):
        raise ValueError("configured sampling frequency is not verified; inference blocked")
    if not MODEL_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError("trained model artifact or model metadata is missing")

    model = TrainedModel(MODEL_PATH, MODEL_METADATA_PATH, CONFIG_PATH)
    reader = Reader()
    try:
        record = reader.get_session(session_id)
        if not record:
            raise ValueError(f"session not found: {session_id}")
        props = record["session"]
        raw_count = int(props.get("raw_eeg_sample_count", 0))
        if raw_count < int(round(fs * float(config["window_duration_seconds"]))):
            raise ValueError("session has too few stored samples for one complete model window")
        labels, samples, timestamps, stored_rate = reader.export_sample_chunks(session_id, raw_count)
    finally:
        reader.close()

    if not math.isclose(stored_rate, fs, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"stored rate {stored_rate:g} Hz differs from model rate {fs:g} Hz")
    mapping = map_eeg_channels(labels, model.channels)
    index_order = [mapping[name] for name in model.channels]
    data = np.asarray(samples, dtype=np.float64)
    stamp_array = np.asarray(timestamps, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(labels) or len(data) != len(stamp_array):
        raise ValueError("stored chunk dimensions or timestamp count are inconsistent")
    if len(data) != raw_count:
        raise ValueError(f"session metadata says {raw_count} raw samples but chunks returned {len(data)}")
    if not np.isfinite(data).all() or not np.isfinite(stamp_array).all():
        raise ValueError("stored EEG contains invalid numeric values or timestamps")
    data = data[:, index_order]

    observed = 1.0 / float(np.median(np.diff(stamp_array))) if len(stamp_array) > 1 else None
    if observed is None or not math.isclose(observed, fs, rel_tol=0.02):
        raise ValueError(f"timestamps imply {observed!r} Hz, incompatible with the {fs:g} Hz model")

    window_samples = int(model.metadata["window_samples"])
    step_samples = int(model.metadata["window_step_samples"])
    buffer = RollingWindowBuffer(len(model.channels), window_samples, step_samples,
                                 expected_rate_hz=fs)
    predictions = []
    # Push in bounded chunks, using the same buffer and TrainedModel implementation
    # as live inference. This preserves chronological order and window alignment.
    for start in range(0, len(data), 512):
        end = min(start + 512, len(data))
        windows = buffer.push(data[start:end], stamp_array[start:end])
        for window, first_ts, last_ts, window_start in windows:
            result = model.predict_window(window, first_ts, last_ts, len(predictions) + 1)
            result["window_start_sample"] = int(window_start)
            predictions.append(result)
    if not predictions:
        raise ValueError("no complete windows were available for prediction")
    return {
        "ok": True,
        "session": {
            "session_id": session_id,
            "started_at_utc": props.get("started_at_utc"),
            "device": props.get("device", "unknown"),
            "sample_count": len(data),
            "duration_seconds": len(data) / fs,
            "sampling_rate_hz": fs,
            "observed_sampling_rate_hz": observed,
            "channel_labels_used": model.channels,
        },
        "model": {
            "algorithm": model.metadata.get("model_algorithm"),
            "model_created_utc": model.metadata.get("model_created_utc"),
            "validation_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy"),
            "validation_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro"),
            "external_validation": bool(model.metadata.get("external_validation", False)),
        },
        "interpretation": "Predictions are baseline/stress experimental-condition estimates, not clinical diagnoses.",
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sessions = commands.add_parser("sessions", help="list stored sessions that have raw EEG")
    sessions.add_argument("--limit", type=int, default=50)
    predict = commands.add_parser("predict-session", help="retrieve and predict one stored session")
    predict.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    try:
        payload = list_sessions(args.limit) if args.command == "sessions" else predict_session(args.session_id)
        print(json.dumps({"ok": True, "sessions": payload} if args.command == "sessions" else payload,
                         separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
