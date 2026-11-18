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
import threading
import time
from datetime import datetime, timezone
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
            "ended_at_utc": session.get("ended_at_utc"),
            "participant_id": session.get("participant_id"),
            "participant_name": session.get("participant_name"),
            "stream_name": session.get("stream_name"),
            "device": session.get("device", "unknown"),
            "channel_count": session.get("channel_count"),
            "nominal_sampling_rate_hz": session.get("nominal_sampling_rate_hz"),
            "observed_sampling_rate_hz": session.get("observed_sampling_rate_hz"),
            "raw_sample_count": int(row.get("raw_sample_count", 0)),
            "raw_chunk_count": int(row.get("raw_chunk_count", 0)),
            "prediction_count_already_stored": int(row.get("prediction_count", 0)),
        })
    return result


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False), flush=True)


def monitor_session(session_id: str, poll_interval: float = 1.0) -> None:
    """Poll one Neo4j chunk per second and incrementally predict complete windows.

    The reader begins at chunk zero, so Device 2 can join after Device 1 has
    already started. Newly committed chunks are consumed in sequence; no EEG
    samples are synthesized, filled, resampled, or modified.
    """
    if not 0.25 <= poll_interval <= 10.0:
        raise ValueError("poll interval must be between 0.25 and 10 seconds")
    RollingWindowBuffer, TrainedModel, map_eeg_channels, Reader = _components()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fs = float(config["sampling_frequency_hz"])
    frequency_verified = bool(config.get("sampling_frequency_verified"))
    model = None
    inference_block_reason = None
    if frequency_verified and MODEL_PATH.is_file() and MODEL_METADATA_PATH.is_file():
        try:
            model = TrainedModel(MODEL_PATH, MODEL_METADATA_PATH, CONFIG_PATH)
        except Exception as exc:
            inference_block_reason = f"Saved model could not be loaded: {exc}"
    elif not frequency_verified:
        inference_block_reason = "Configured model sampling frequency is not verified."
    else:
        inference_block_reason = "Trained model artifact or metadata is missing."
    reader = Reader()
    stop_event = threading.Event()

    def watch_stop() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                continue
            if command.get("command") == "stop":
                stop_event.set()
                return

    threading.Thread(target=watch_stop, name="device2-stop-listener", daemon=True).start()
    sample_total = 0
    chunk_cursor = 0
    window_count = 0
    previous_timestamp: float | None = None
    prediction_rows: list[dict[str, Any]] = []
    monitor_started = datetime.now(timezone.utc).isoformat()
    try:
        props = reader.get_session_properties(session_id)
        if not props:
            raise ValueError(f"session not found: {session_id}")
        labels = list(props.get("channel_labels") or [])
        if not labels or int(props.get("channel_count", len(labels))) != len(labels):
            raise ValueError("session channel metadata is incomplete; cannot safely map EEG channels")
        nominal_rate = float(props.get("nominal_sampling_rate_hz") or 0.0)
        selected_indices = list(range(len(labels)))
        window_buffer = None
        window_samples = 0
        step_samples = 0
        if model is not None:
            try:
                mapping = map_eeg_channels(labels, model.channels)
                if nominal_rate <= 0 or not math.isclose(nominal_rate, fs, rel_tol=0.02):
                    raise ValueError(f"session nominal rate {nominal_rate:g} Hz is incompatible with model rate {fs:g} Hz")
                selected_indices = [mapping[name] for name in model.channels]
                window_samples = int(model.metadata["window_samples"])
                step_samples = int(model.metadata["window_step_samples"])
                window_buffer = RollingWindowBuffer(len(model.channels), window_samples, step_samples,
                                                    expected_rate_hz=fs)
            except (ValueError, KeyError, TypeError) as exc:
                inference_block_reason = str(exc)
                model = None
        display_labels = labels if labels and any(label.strip() for label in labels) else [
            f"Ch {index + 1:02d}" for index in range(len(labels))]
        _emit({"type": "monitor_started", "session": {
            "session_id": session_id,
            "participant_id": props.get("participant_id"),
            "participant_name": props.get("participant_name"),
            "started_at_utc": props.get("started_at_utc"),
            "device": props.get("device", "unknown"),
            "stream_name": props.get("stream_name", "unknown"),
            "channel_count": len(labels),
            "channel_labels": display_labels,
            "channel_labels_used": model.channels if model is not None else [],
            "nominal_sampling_rate_hz": nominal_rate,
        }, "inference_enabled": model is not None,
            "inference_block_reason": inference_block_reason,
            "model": ({"algorithm": model.metadata.get("model_algorithm"),
                    "window_samples": window_samples, "window_step_samples": step_samples,
                    "window_duration_seconds": window_samples / fs,
                    "validation_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy"),
                    "validation_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro"),
                    "external_validation": bool(model.metadata.get("external_validation", False))}
                    if model is not None else None),
            "poll_interval_seconds": poll_interval,
            "message": ("Reading stored EEG chunks from the beginning of this session."
                        if model is not None else
                        "Reading and graphing stored raw channels; model predictions are blocked until channel/rate metadata is valid.")})

        while not stop_event.is_set():
            cycle_started = time.monotonic()
            chunks = reader.read_sample_chunks_after(session_id, chunk_cursor, limit=10)
            for chunk in chunks:
                if chunk["chunk_index"] != chunk_cursor:
                    raise ValueError(f"chunk sequence gap: expected {chunk_cursor}, got {chunk['chunk_index']}")
                if chunk["channel_labels"] != labels:
                    raise ValueError(f"channel labels/order changed at chunk {chunk_cursor}")
                chunk_rate = float(chunk["sampling_rate_hz"])
                if chunk_rate <= 0:
                    raise ValueError(f"chunk {chunk_cursor} has invalid sampling rate {chunk_rate:g} Hz")
                if model is not None and not math.isclose(chunk_rate, fs, rel_tol=0.02):
                    raise ValueError(f"chunk {chunk_cursor} rate {chunk['sampling_rate_hz']:g} Hz differs from model rate")
                # One complete second is the normal live unit. A shorter final
                # chunk is accepted only after Device 1 has marked the session ended.
                current_props = reader.get_session_properties(session_id) or props
                n = chunk["sample_count"]
                expected_chunk_samples = int(round(chunk_rate))
                if n != expected_chunk_samples and not (current_props.get("ended_at_utc") and 0 < n < expected_chunk_samples):
                    raise ValueError(f"chunk {chunk_cursor} contains {n} samples; expected {expected_chunk_samples} per second")
                raw = np.asarray(chunk["samples"], dtype=np.float64)
                stamps = np.asarray(chunk["timestamps"], dtype=np.float64)
                if raw.shape != (n, len(labels)) or not np.isfinite(raw).all() or not np.isfinite(stamps).all():
                    raise ValueError(f"chunk {chunk_cursor} contains malformed or non-finite samples/timestamps")
                if len(stamps) > 1:
                    intervals = np.diff(stamps)
                    if np.any(intervals <= 0):
                        raise ValueError(f"chunk {chunk_cursor} has non-increasing LSL timestamps")
                    median_interval = float(np.median(intervals))
                    observed_rate = 1.0 / median_interval
                    if np.any(intervals > 1.5 * median_interval):
                        raise ValueError(f"chunk {chunk_cursor} timestamps contain an irregular interval or gap")
                    if model is not None and not math.isclose(observed_rate, fs, rel_tol=0.02):
                        raise ValueError(f"chunk {chunk_cursor} timestamps are incompatible with model rate {fs:g} Hz")
                if previous_timestamp is not None:
                    boundary = float(stamps[0]) - previous_timestamp
                    if boundary <= 0 or (model is not None and boundary > 1.5 / fs):
                        raise ValueError(f"timestamp gap at boundary before chunk {chunk_cursor}")
                if int(chunk["first_sample_index"]) != sample_total:
                    raise ValueError(f"sample-index gap at chunk {chunk_cursor}: expected {sample_total}, got {chunk['first_sample_index']}")

                eeg = raw[:, selected_indices]
                windows = window_buffer.push(eeg, stamps) if window_buffer is not None else []
                sample_offset = sample_total
                sample_total += n
                previous_timestamp = float(stamps[-1])
                chunk_cursor += 1
                _emit({"type": "sample_chunk", "chunk_index": chunk["chunk_index"],
                       "first_sample_index": sample_offset, "sample_count": n,
                       "timestamps": stamps.tolist(), "samples": eeg.tolist(),
                       "channel_labels": display_labels if model is None else model.channels,
                       "samples_processed": sample_total,
                       "elapsed_seconds": sample_total / fs,
                       "observed_rate_hz": observed_rate if n > 1 else None})
                for window, first_ts, last_ts, window_start in windows:
                    window_count += 1
                    result = model.predict_window(window, first_ts, last_ts, window_count)
                    result["window_start_sample"] = int(window_start)
                    result["elapsed_seconds"] = (int(window_start) + window_samples) / fs
                    prediction_rows.append(result)
                    _emit({"type": "prediction", "prediction": result,
                           "prediction_count": window_count})

            props = reader.get_session_properties(session_id) or props
            expected_chunks = int(props.get("raw_eeg_chunk_count", 0) or 0)
            if props.get("ended_at_utc") and chunk_cursor >= expected_chunks:
                break
            if len(chunks) < 10:
                # Poll on a one-second cadence. If a database response or model
                # calculation is slow, skip the sleep and catch up on the next
                # query rather than allowing an avoidable backlog to grow.
                stop_event.wait(max(0.0, poll_interval - (time.monotonic() - cycle_started)))
    finally:
        reader.close()

    probabilities = [float(row["stress_probability"]) for row in prediction_rows
                     if row.get("stress_probability") is not None and math.isfinite(float(row["stress_probability"]))]
    stress_count = sum(1 for row in prediction_rows if row.get("predicted_condition") == "stress")
    baseline_count = sum(1 for row in prediction_rows if row.get("predicted_condition") == "baseline")
    if stop_event.is_set():
        finish_reason = "stopped_by_user"
    else:
        finish_reason = "device1_session_ended"
    _emit({"type": "monitor_stopped", "session_id": session_id,
           "started_at_utc": monitor_started,
           "ended_at_utc": datetime.now(timezone.utc).isoformat(),
           "finish_reason": finish_reason,
           "sample_count": sample_total,
           "processed_duration_seconds": sample_total / fs,
           "chunk_count": chunk_cursor,
           "window_count": window_count,
           "stress_window_count": stress_count,
           "baseline_window_count": baseline_count,
           "average_stress_score": float(np.mean(probabilities)) if probabilities else None,
           "inference_enabled": model is not None,
           "inference_block_reason": inference_block_reason,
           "majority_window_condition": ("stress" if stress_count > baseline_count else
                                         "baseline" if baseline_count > stress_count else "tie_or_no_windows"),
           "interpretation": "Experimental-condition estimates; not a clinical diagnosis."})


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
    monitor = commands.add_parser("monitor-session", help="poll one stored session and predict as new 1 s chunks arrive")
    monitor.add_argument("--session-id", required=True)
    monitor.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "monitor-session":
            monitor_session(args.session_id, args.poll_interval)
            return 0
        payload = list_sessions(args.limit) if args.command == "sessions" else predict_session(args.session_id)
        print(json.dumps({"ok": True, "sessions": payload} if args.command == "sessions" else payload,
                         separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
