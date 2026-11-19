"""Discover EEGStudio LSL streams, record them, and run trained-model inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eeg_pipeline import (RollingWindowBuffer, TrainedModel, estimate_rate,
                          extract_stream_metadata, map_eeg_channels)
from eeg_recording import LocalRecording

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "eeg_config.json"
DEFAULT_MODEL = ROOT / "stress_model.pkl"
DEFAULT_MODEL_METADATA = ROOT / "model_metadata.json"


def resolve_streams(wait_seconds: float = 2.0) -> list[Any]:
    try:
        from pylsl import resolve_streams
    except ImportError as exc:
        raise RuntimeError("pylsl is not installed; install this folder's requirements.txt") from exc
    return list(resolve_streams(wait_time=wait_seconds))


def print_streams(streams: list[Any]) -> None:
    if not streams:
        print("No LSL streams found. Start EEGStudio's LSL Outlet or a test publisher first.")
        return
    for index, info in enumerate(streams):
        meta = extract_stream_metadata(info)
        print(f"[{index}] uid={meta.stream_id!r} name={meta.name!r} type={meta.stream_type!r} source_id={meta.source_id!r} "
              f"channels={meta.channel_count} nominal_hz={meta.nominal_srate:g} "
              f"format={meta.channel_format} labels={meta.channel_labels}")
        if not meta.channel_labels or not any(meta.channel_labels):
            print("    Resolver metadata is partial; channel labels are read from the connected inlet during inspect/record.")
        if "marker" in meta.stream_type.casefold() or "event" in meta.stream_type.casefold():
            print("    Stream appears to be a marker/event stream based on metadata; verify with EEGStudio.")
        if meta.source_id.startswith("RECORDED_CSV_"):
            print("    This is a recorded CSV replay stream, not live EEG hardware.")


def _emit_json(enabled: bool, event_type: str, **payload: Any) -> None:
    if enabled:
        print(json.dumps({"type": event_type, **payload}, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False), flush=True)


def choose_stream(streams: list[Any], index: int | None, stream_uid: str | None = None) -> Any:
    if not streams:
        raise RuntimeError("no LSL streams available")
    if stream_uid:
        matches = [item for item in streams if extract_stream_metadata(item).stream_id == stream_uid]
        if not matches:
            raise ValueError(f"LSL stream UID {stream_uid!r} is no longer available; run discover again")
        # Resolver duplicates can represent the same outlet on multiple
        # interfaces. An exact UID identifies the same outlet instance.
        return matches[0]
    if index is None:
        if len(streams) == 1:
            raise RuntimeError("one stream is available but automatic selection is disabled; pass --stream-index 0")
        raise RuntimeError("multiple streams available; select one explicitly with --stream-index")
    if index < 0 or index >= len(streams):
        raise ValueError(f"stream index {index} is outside 0..{len(streams)-1}")
    return streams[index]


def connect(info: Any, timeout: float) -> Any:
    try:
        from pylsl import StreamInlet
    except ImportError as exc:
        raise RuntimeError("pylsl is not installed; install this folder's requirements.txt") from exc
    # No timestamp-processing flags: persist the LSL timestamp exactly as received.
    inlet = StreamInlet(info, max_buflen=60, processing_flags=0)
    inlet.open_stream(timeout=timeout)
    return inlet


def _run(args: argparse.Namespace, infer: bool) -> int:
    json_events = bool(getattr(args, "json_events", False))
    inference_enabled = infer
    inference_block_reason = None
    config = json.loads(args.config.read_text(encoding="utf-8"))
    participant_id = (getattr(args, "participant_id", None) or "").strip()
    participant_name = (getattr(args, "participant_name", None) or "").strip()
    if len(participant_id) > 100 or len(participant_name) > 160:
        raise ValueError("participant ID must be <=100 characters and display name <=160 characters")
    participant_properties = {}
    if participant_id:
        participant_properties["participant_id"] = participant_id
    if participant_name:
        participant_properties["participant_name"] = participant_name
    streams = resolve_streams(args.resolve_wait)
    if not json_events:
        print_streams(streams)
    info = choose_stream(streams, args.stream_index, getattr(args, "stream_uid", None))
    if infer:
        try:
            model = TrainedModel(args.model, args.model_metadata, args.config)
            expected = float(model.metadata["sampling_frequency_hz"])
            window_samples = int(model.metadata["window_samples"])
            step_samples = int(model.metadata["window_step_samples"])
            model_indices = []
            buffer = None
        except Exception as exc:
            if not getattr(args, "allow_unmapped", False):
                raise
            model = None; expected = None; window_samples = 0; step_samples = 0
            model_indices = []; buffer = None
            inference_enabled = False
            inference_block_reason = f"Saved model could not be loaded: {exc}"
    else:
        model = None; model_indices = []; expected = None; buffer = None

    try:
        inlet = connect(info, args.connect_timeout)
    except Exception:
        raise
    # Resolver StreamInfo descriptions are partial; get channel metadata after connection.
    meta = extract_stream_metadata(inlet.info())
    raw_channel_labels = list(meta.channel_labels)
    try:
        if len(raw_channel_labels) > meta.channel_count:
            raise RuntimeError("stream metadata has more channel labels than samples; refusing to record")
        # Unlabelled streams can still be preserved and plotted using neutral column
        # names. These generated names are never used to map model EEG channels.
        channel_labels = [str(label or "").strip() for label in raw_channel_labels]
        channel_labels.extend([""] * (meta.channel_count - len(channel_labels)))
        channel_labels = [label or f"Ch {index + 1:02d}" for index, label in enumerate(channel_labels)]
        if inference_enabled:
            try:
                channel_map = map_eeg_channels(raw_channel_labels, model.channels)
                if meta.nominal_srate <= 0 or not np.isclose(meta.nominal_srate, expected, rtol=args.rate_tolerance):
                    raise ValueError(f"nominal stream rate {meta.nominal_srate:g} Hz is incompatible with model rate {expected:g} Hz")
                model_indices = [channel_map[channel] for channel in model.channels]
                buffer = RollingWindowBuffer(len(model.channels), window_samples, step_samples,
                                             expected_rate_hz=expected, rate_tolerance=args.rate_tolerance)
            except Exception as exc:
                if not getattr(args, "allow_unmapped", False):
                    raise
                inference_enabled = False
                inference_block_reason = str(exc)
                model = None
                model_indices = list(range(meta.channel_count))
                buffer = None
        else:
            model_indices = list(range(meta.channel_count))
    except Exception:
        inlet.close_stream()
        raise
    event_inlet = None
    event_meta = None
    if args.event_stream_index is not None:
        event_info = choose_stream(streams, args.event_stream_index)
        if extract_stream_metadata(event_info).stream_id == meta.stream_id:
            inlet.close_stream()
            raise ValueError("event stream must be selected separately from the EEG stream")
        try:
            event_inlet = connect(event_info, args.connect_timeout)
            event_meta = extract_stream_metadata(event_inlet.info())
        except Exception:
            inlet.close_stream()
            raise
    session_id = str(uuid.uuid4())
    recorder = LocalRecording(args.output, session_id, channel_labels)
    log_path = args.output / "logs" / f"acquisition_{session_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    session_logger = logging.getLogger(f"eeg_acquisition.{session_id}")
    session_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    session_logger.addHandler(file_handler)
    start_wall = datetime.now(timezone.utc).isoformat()
    timestamp_history: deque[float] = deque(maxlen=10000)
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    timestamp_count = 0
    last_data_time = time.monotonic()
    empty_chunk_count = 0
    interruption_count = 0
    previous_chunk_last: float | None = None
    window_count = 0
    prediction_history: list[dict[str, Any]] = []
    session_record = {"session_id": session_id, "stream_name": meta.name, "stream_type": meta.stream_type,
        "source_id": meta.source_id, "properties": {"started_at_utc": start_wall,
        **participant_properties,
        "stream_name": meta.name, "stream_type": meta.stream_type, "source_id": meta.source_id,
        "channel_count": meta.channel_count, "channel_labels": channel_labels,
        "channel_labels_raw": raw_channel_labels,
        "nominal_sampling_rate_hz": meta.nominal_srate, "channel_format": meta.channel_format,
        "model_algorithm": model.metadata.get("model_algorithm") if model else None,
        "model_validation_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy") if model else None,
        "model_validation_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro") if model else None,
        "channel_labels_used": model.channels if model else [],
        "prediction_enabled": inference_enabled,
        "inference_block_reason": inference_block_reason,
        "signal_units": "unknown", "device": ("simulated LSL test publisher; not Mitsar hardware"
            if meta.source_id.startswith("SIMULATED_") else "recorded CSV replay; not live hardware"
            if meta.source_id.startswith("RECORDED_CSV_") else "Mitsar SmartBCI x24 (reported; LSL source unverified)"),
        "eeg_csv_path": str(recorder.csv_path)}}
    _emit_json(json_events, "session_started", session={"session_id": session_id,
        **participant_properties, "stream_name": meta.name, "stream_type": meta.stream_type,
        "source_id": meta.source_id, "channel_count": meta.channel_count,
        "channel_labels": channel_labels, "channel_labels_raw": raw_channel_labels,
        "nominal_sampling_rate_hz": meta.nominal_srate,
        "started_at_utc": start_wall, "device": session_record["properties"]["device"]},
        inference_enabled=inference_enabled, inference_block_reason=inference_block_reason,
        model={"algorithm": model.metadata.get("model_algorithm"),
               "validation_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy"),
               "validation_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro")} if model else None)
    if not json_events:
        print(f"Session ID: {session_id}", flush=True)
    base_metadata = {"session_id": session_id, "stream_name": meta.name, "stream_type": meta.stream_type,
        "source_id": meta.source_id, "stream_uid": meta.stream_id, "channel_count": meta.channel_count,
        **participant_properties,
        "channel_labels_in_stream_order": channel_labels, "channel_labels_raw": raw_channel_labels,
        "nominal_sampling_rate_hz": meta.nominal_srate,
        "channel_format": meta.channel_format, "recording_start_utc": start_wall,
        "log_path": str(log_path),
        "signal_units": "unknown", "data_type": "preserved values as received; LSL channel format above",
        "device_context": ({"source": "synthetic test publisher", "hardware_test": False}
            if meta.source_id.startswith("SIMULATED_") else
            {"source": "raw CSV replay", "hardware_test": False}
            if meta.source_id.startswith("RECORDED_CSV_") else
            {"manufacturer": "Mitsar", "model": "SmartBCI x24", "serial": "6952015"}),
        "selected_eeg_channel_indices": ({name: index for name, index in zip(model.channels, model_indices)} if model else None),
        "event_stream": ({"name": event_meta.name, "type": event_meta.stream_type,
            "source_id": event_meta.source_id, "channel_count": event_meta.channel_count,
            "channel_labels": event_meta.channel_labels} if event_meta else None),
        "prediction_enabled": inference_enabled,
        "inference_block_reason": inference_block_reason,
        "unit_warning": "Physical scale and raw/processed status are unverified."}
    started = time.monotonic()
    if not json_events:
        print(f"Connected to {meta.name!r}; recording to {recorder.csv_path}. Ctrl+C to stop safely.")
    else:
        _emit_json(True, "status", message="Connected to EEG stream; live inference is running.")
    session_logger.info("Connected to stream=%s source_id=%s channels=%d nominal_rate=%s",
                        meta.name, meta.source_id, meta.channel_count, meta.nominal_srate)
    if model and not json_events:
        print("WARNING: exploratory condition predictions only; model has no external validation and signal units/raw status remain unverified.")
    elif model:
        _emit_json(True, "notice", message="Experimental-condition estimates only; not a clinical diagnosis. Signal units and raw/processed status remain unverified.")
    elif inference_block_reason and json_events:
        _emit_json(True, "notice", message=f"Raw acquisition continues, but predictions are disabled: {inference_block_reason}")
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            if args.stop_file and args.stop_file.is_file():
                if not json_events:
                    print("Stop requested by UI; flushing recording.")
                break
            chunk, timestamps = inlet.pull_chunk(timeout=0.25, max_samples=512)
            if not chunk:
                empty_chunk_count += 1
                if time.monotonic() - last_data_time > args.disconnect_timeout:
                    raise TimeoutError(f"no EEG samples received for {args.disconnect_timeout:g} seconds; connection interrupted")
                continue
            last_data_time = time.monotonic()
            stamp_values = [float(value) for value in timestamps]
            if len(stamp_values) != len(chunk):
                raise RuntimeError("LSL sample/timestamp count mismatch; refusing incomplete recording chunk")
            if stamp_values:
                if first_timestamp is None:
                    first_timestamp = stamp_values[0]
                if previous_chunk_last is not None and stamp_values[0] - previous_chunk_last > 1.5 / max(meta.nominal_srate, 1.0):
                    interruption_count += 1
                previous_chunk_last = stamp_values[-1]
                last_timestamp = stamp_values[-1]
                timestamp_count += len(stamp_values)
                timestamp_history.extend(stamp_values)
            recorder.write_samples(chunk, stamp_values)
            if json_events:
                try:
                    display_values_valid = np.isfinite(np.asarray(chunk, dtype=float)).all() and np.isfinite(stamp_values).all()
                except (TypeError, ValueError):
                    display_values_valid = False
                if display_values_valid:
                    _emit_json(True, "sample_chunk", first_sample_index=recorder.sample_count - len(chunk),
                               sample_count=len(chunk), samples=chunk, timestamps=stamp_values)
                else:
                    _emit_json(True, "signal_quality_warning", message="Non-numeric or non-finite sample/timestamp chunk received; preserved locally, not drawn or used for model inference.")
            if event_inlet is not None:
                event_samples, event_timestamps = event_inlet.pull_chunk(timeout=0.0, max_samples=256)
                if event_samples:
                    recorder.write_events(event_samples, event_timestamps)
            if not inference_enabled:
                continue
            eeg = np.asarray(chunk, dtype=float)[:, model_indices]
            try:
                windows = buffer.push(eeg, stamp_values)
            except ValueError as exc:
                if not getattr(args, "allow_unmapped", False):
                    raise
                inference_enabled = False
                inference_block_reason = f"Incoming window validation failed: {exc}"
                session_record["properties"].update({"prediction_enabled": False,
                    "inference_block_reason": inference_block_reason})
                _emit_json(json_events, "inference_blocked", reason=inference_block_reason)
                continue
            for signal_window, t0, t1, start_sample in windows:
                rate_result = estimate_rate(timestamp_history, meta.nominal_srate, expected, args.rate_tolerance)
                if not rate_result["compatible"]:
                    if not getattr(args, "allow_unmapped", False):
                        raise RuntimeError(f"timestamp-derived rate is not compatible with model: {rate_result}")
                    inference_enabled = False
                    inference_block_reason = f"Timestamp-derived sampling rate is incompatible with the trained model: {rate_result}"
                    session_record["properties"].update({"prediction_enabled": False,
                        "inference_block_reason": inference_block_reason})
                    _emit_json(json_events, "inference_blocked", reason=inference_block_reason)
                    break
                window_count += 1
                result = model.predict_window(signal_window, t0, t1, window_count)
                result["window_start_sample"] = start_sample
                result["elapsed_seconds"] = (start_sample + window_samples) / expected
                recorder.write_prediction(result)
                prediction_history.append(result)
                if json_events:
                    _emit_json(True, "prediction", prediction=result)
                else:
                    print(f"window {window_count}: {result['prediction_text']} "
                          f"(baseline={result.get('baseline_probability', float('nan')):.3f}, "
                          f"stress={result.get('stress_probability', float('nan')):.3f})")
                session_logger.info("window=%d prediction=%s baseline_probability=%s stress_probability=%s",
                    window_count, result["predicted_condition"], result.get("baseline_probability"),
                    result.get("stress_probability"))
    except KeyboardInterrupt:
        if not json_events:
            print("Stop requested; flushing local recording.")
    finally:
        end_wall = datetime.now(timezone.utc).isoformat()
        session_logger.info("Stopping session; samples=%d windows=%d", recorder.sample_count, window_count)
        rate_result = estimate_rate(timestamp_history, meta.nominal_srate, expected or 0.0, args.rate_tolerance)
        stress_scores = [float(row["stress_probability"]) for row in prediction_history
                         if row.get("stress_probability") is not None]
        stress_windows = sum(row.get("predicted_condition") == "stress" for row in prediction_history)
        baseline_windows = sum(row.get("predicted_condition") == "baseline" for row in prediction_history)
        majority_condition = ("stress" if stress_windows > baseline_windows else
                              "baseline" if baseline_windows > stress_windows else "tie_or_no_windows")
        average_stress_score = float(np.mean(stress_scores)) if stress_scores else None
        final_metadata = {**base_metadata, "recording_end_utc": end_wall,
            "sample_count": recorder.sample_count,
            "timestamp_count": timestamp_count,
            "recording_duration_seconds": (last_timestamp - first_timestamp) if first_timestamp is not None and last_timestamp is not None else 0,
            "timestamp_diagnostics_sample_count": len(timestamp_history),
            "empty_chunk_count": empty_chunk_count, "detected_chunk_interruption_count": interruption_count,
            "event_sample_count": recorder.event_count,
            "sampling_rate_validation": rate_result, "window_count": window_count,
            "model_algorithm": model.metadata.get("model_algorithm") if model else None,
            "prediction_enabled": inference_enabled,
            "inference_block_reason": inference_block_reason,
            "model_cv_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy") if model else None,
            "model_cv_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro") if model else None,
            "prediction_interpretation": "experimental condition, not clinical diagnosis" if model else None,
            "event_stream_status": "recorded separately" if event_inlet else "not selected; inspect discover output and pass --event-stream-index if present",
            "events_csv_path": str(recorder.events_path) if event_inlet else None,
            "predictions_csv_path": str(recorder.predictions_path),
            "participant": participant_properties,
            "summary": {"sample_count": recorder.sample_count, "window_count": window_count,
                "recording_duration_seconds": (last_timestamp - first_timestamp) if first_timestamp is not None and last_timestamp is not None else 0,
                "observed_sampling_rate_hz": rate_result["observed_rate_hz"],
                "stress_window_count": int(stress_windows), "baseline_window_count": int(baseline_windows),
                "average_stress_score": average_stress_score, "majority_window_condition": majority_condition},
            "log_path": str(log_path)}
        recorder.write_metadata(final_metadata)
        recorder.close()
        try:
            inlet.close_stream()
            if event_inlet:
                event_inlet.close_stream()
        except Exception:
            logging.exception("Error while closing LSL inlet")
        final_metadata["recording_end_utc"] = end_wall
        final_metadata["summary"] = {"sample_count": recorder.sample_count, "window_count": window_count,
            "recording_duration_seconds": (last_timestamp - first_timestamp) if first_timestamp is not None and last_timestamp is not None else 0,
            "observed_sampling_rate_hz": rate_result["observed_rate_hz"],
            "stress_window_count": int(stress_windows), "baseline_window_count": int(baseline_windows),
            "average_stress_score": average_stress_score, "majority_window_condition": majority_condition}
        recorder.write_metadata(final_metadata)
        summary = {"sample_count": recorder.sample_count, "window_count": window_count,
            "duration_seconds": (last_timestamp - first_timestamp) if first_timestamp is not None and last_timestamp is not None else 0,
            "observed_sampling_rate_hz": rate_result["observed_rate_hz"],
            "stress_window_count": int(stress_windows), "baseline_window_count": int(baseline_windows),
            "average_stress_score": average_stress_score,
            "majority_window_condition": majority_condition,
            "local_recording": str(recorder.csv_path), "metadata_file": str(recorder.metadata_path),
            "predictions_file": str(recorder.predictions_path)}
        if json_events:
            _emit_json(True, "session_stopped", session={"session_id": session_id,
                **participant_properties, "stream_name": meta.name, "channel_count": meta.channel_count,
            "channel_labels": channel_labels, "channel_labels_raw": raw_channel_labels,
            "channel_labels_used": model.channels if model else [],
                "nominal_sampling_rate_hz": meta.nominal_srate, "started_at_utc": start_wall,
                "ended_at_utc": end_wall}, summary=summary,
                inference_enabled=inference_enabled, inference_block_reason=inference_block_reason,
                model={"algorithm": model.metadata.get("model_algorithm"),
                    "validation_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy"),
                    "validation_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro")} if model else None,
                predictions=prediction_history)
        else:
            print(f"Saved {recorder.sample_count} samples, {window_count} predictions; observed rate="
                  f"{rate_result['observed_rate_hz']} Hz; files: {recorder.csv_path}, {recorder.metadata_path}, {recorder.predictions_path}")
            print(f"Session ID: {session_id}")
        session_logger.removeHandler(file_handler)
        file_handler.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="list all visible LSL streams")
    discover.add_argument("--resolve-wait", type=float, default=2.0)
    discover.add_argument("--json", action="store_true", help="emit stream metadata as JSON")
    inspect = sub.add_parser("inspect", help="inspect metadata for a selected stream")
    inspect.add_argument("--stream-index", type=int, required=True)
    inspect.add_argument("--resolve-wait", type=float, default=2.0)
    for name, help_text in (("record", "record original LSL samples locally"),
                            ("predict", "record and run trained-model window predictions"),
                            ("live", "stream predictions continuously")):
        cmd = sub.add_parser(name, help=help_text)
        stream_selection = cmd.add_mutually_exclusive_group(required=True)
        stream_selection.add_argument("--stream-index", type=int, help="explicit index from discover output")
        stream_selection.add_argument("--stream-uid", help="select by exact LSL stream UID from discovery metadata")
        cmd.add_argument("--event-stream-index", type=int, help="optional separate marker/event stream index")
        cmd.add_argument("--duration", type=float, help="seconds; omit to run until Ctrl+C")
        cmd.add_argument("--resolve-wait", type=float, default=2.0)
        cmd.add_argument("--connect-timeout", type=float, default=10.0)
        cmd.add_argument("--disconnect-timeout", type=float, default=10.0,
                         help="stop and flush if no EEG samples arrive within this many seconds")
        cmd.add_argument("--rate-tolerance", type=float, default=0.02)
        cmd.add_argument("--participant-id", help="study participant code stored with the session")
        cmd.add_argument("--participant-name", help="participant name saved with the session for the desktop archive")
        cmd.add_argument("--stop-file", type=Path,
                         help="stop recording gracefully when this file appears; used by the Electron desktop UI")
        cmd.add_argument("--output", type=Path, default=ROOT / "data" / "live")
        cmd.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        cmd.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        cmd.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
        cmd.add_argument("--json-events", action="store_true", help="emit JSON events for the Electron live interface")
        cmd.add_argument("--allow-unmapped", action="store_true",
                         help="continue raw local acquisition but disable inference when model input validation fails")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            streams = resolve_streams(args.resolve_wait)
            if args.json:
                model_config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
                print(json.dumps({"ok": True,
                    "model_input": {"channels": model_config["eeg_channels"],
                        "sampling_frequency_hz": model_config["sampling_frequency_hz"]},
                    "streams": [asdict(extract_stream_metadata(info)) for info in streams]},
                    separators=(",", ":"), ensure_ascii=False), flush=True)
            else:
                print_streams(streams)
            return 0
        if args.command == "inspect":
            streams = resolve_streams(args.resolve_wait); print_streams(streams)
            selected = choose_stream(streams, args.stream_index)
            inlet = connect(selected, 10.0)
            try:
                print(json.dumps(extract_stream_metadata(inlet.info()).__dict__, indent=2))
            finally:
                inlet.close_stream()
            return 0
        return _run(args, infer=args.command in {"predict", "live"})
    except Exception as exc:
        if getattr(args, "json_events", False):
            print(json.dumps({"type": "error", "error": str(exc)}, separators=(",", ":")), flush=True)
            return 2
        parser.exit(2, f"Acquisition stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
