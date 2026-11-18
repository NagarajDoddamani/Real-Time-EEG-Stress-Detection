"""Discover EEGStudio LSL streams, record them, and run trained-model inference."""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.acquisition.pipeline import (RollingWindowBuffer, TrainedModel, estimate_rate,
                                      extract_stream_metadata, map_eeg_channels)
from src.acquisition.recording import LocalRecording

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "eeg_config.json"
DEFAULT_MODEL = ROOT / "models" / "stress_model.pkl"
DEFAULT_MODEL_METADATA = ROOT / "models" / "model_metadata.json"


def resolve_streams(wait_seconds: float = 2.0) -> list[Any]:
    try:
        from pylsl import resolve_streams
    except ImportError as exc:
        raise RuntimeError("pylsl is not installed; install requirements.txt") from exc
    return list(resolve_streams(wait_time=wait_seconds))


def print_streams(streams: list[Any]) -> None:
    if not streams:
        print("No LSL streams found. Start EEGStudio's LSL Outlet or a test publisher first.")
        return
    for index, info in enumerate(streams):
        meta = extract_stream_metadata(info)
        print(f"[{index}] name={meta.name!r} type={meta.stream_type!r} source_id={meta.source_id!r} "
              f"channels={meta.channel_count} nominal_hz={meta.nominal_srate:g} "
              f"format={meta.channel_format} labels={meta.channel_labels}")
        if not meta.channel_labels or not any(meta.channel_labels):
            print("    Resolver metadata is partial; channel labels are read from the connected inlet during inspect/record.")
        if "marker" in meta.stream_type.casefold() or "event" in meta.stream_type.casefold():
            print("    Stream appears to be a marker/event stream based on metadata; verify with EEGStudio.")
        if meta.source_id.startswith("RECORDED_CSV_"):
            print("    This is a recorded CSV replay stream, not live EEG hardware.")


def choose_stream(streams: list[Any], index: int | None) -> Any:
    if not streams:
        raise RuntimeError("no LSL streams available")
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
        raise RuntimeError("pylsl is not installed; install requirements.txt") from exc
    # No timestamp-processing flags: persist the LSL timestamp exactly as received.
    inlet = StreamInlet(info, max_buflen=60, processing_flags=0)
    inlet.open_stream(timeout=timeout)
    return inlet


def _run(args: argparse.Namespace, infer: bool) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    streams = resolve_streams(args.resolve_wait)
    print_streams(streams)
    info = choose_stream(streams, args.stream_index)
    if infer:
        model = TrainedModel(args.model, args.model_metadata, args.config)
        expected = float(model.metadata["sampling_frequency_hz"])
        window_samples = int(model.metadata["window_samples"])
        step_samples = int(model.metadata["window_step_samples"])
        model_indices = []
        buffer = None
    else:
        model = None; model_indices = []; expected = None; buffer = None

    neo4j = None
    if args.neo4j:
        from src.storage.neo4j_store import Neo4jWriter
        neo4j = Neo4jWriter()
    try:
        inlet = connect(info, args.connect_timeout)
    except Exception:
        if neo4j:
            neo4j.close()
        raise
    # Resolver StreamInfo descriptions are partial; get channel metadata after connection.
    meta = extract_stream_metadata(inlet.info())
    try:
        if meta.channel_count != len(meta.channel_labels):
            raise RuntimeError("stream metadata channel count and labels differ; refusing to record or infer")
        if infer:
            channel_map = map_eeg_channels(meta.channel_labels, model.channels)
            if meta.nominal_srate > 0 and not np.isclose(meta.nominal_srate, expected, rtol=args.rate_tolerance):
                raise RuntimeError(f"nominal stream rate {meta.nominal_srate:g} Hz is incompatible with model rate {expected:g} Hz")
            model_indices = [channel_map[channel] for channel in model.channels]
            buffer = RollingWindowBuffer(len(model.channels), window_samples, step_samples,
                                         expected_rate_hz=expected, rate_tolerance=args.rate_tolerance)
        else:
            model_indices = list(range(meta.channel_count))
    except Exception:
        inlet.close_stream()
        if neo4j:
            neo4j.close()
        raise
    db_raw_seconds = float(getattr(args, "db_raw_seconds", 0.0))
    if db_raw_seconds < 0:
        inlet.close_stream()
        if neo4j: neo4j.close()
        raise ValueError("--db-raw-seconds cannot be negative")
    db_raw_sample_limit = 0
    db_storage_rate = None
    if db_raw_seconds > 0:
        if not neo4j:
            inlet.close_stream()
            raise ValueError("--db-raw-seconds requires --neo4j")
        storage_rate = expected or meta.nominal_srate
        if storage_rate <= 0 or not np.isclose(storage_rate, 500.0, rtol=args.rate_tolerance):
            inlet.close_stream(); neo4j.close()
            raise ValueError("bounded raw-to-DB test requires an advertised/model-compatible 500 Hz stream")
        db_storage_rate = float(storage_rate)
        db_raw_sample_limit = int(round(storage_rate * db_raw_seconds))
    event_inlet = None
    event_meta = None
    if args.event_stream_index is not None:
        event_info = choose_stream(streams, args.event_stream_index)
        if extract_stream_metadata(event_info).stream_id == meta.stream_id:
            inlet.close_stream()
            if neo4j: neo4j.close()
            raise ValueError("event stream must be selected separately from the EEG stream")
        try:
            event_inlet = connect(event_info, args.connect_timeout)
            event_meta = extract_stream_metadata(event_inlet.info())
        except Exception:
            inlet.close_stream()
            if neo4j:
                neo4j.close()
            raise
    session_id = str(uuid.uuid4())
    recorder = LocalRecording(args.output, session_id, meta.channel_labels)
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
    raw_chunk_collector = None
    if db_raw_sample_limit:
        from src.storage.neo4j_store import SampleChunkCollector
        raw_chunk_collector = SampleChunkCollector(neo4j, session_id, meta.channel_labels,
            db_storage_rate, db_raw_sample_limit, chunk_samples=int(round(db_storage_rate)))
    session_record = {"session_id": session_id, "stream_name": meta.name, "stream_type": meta.stream_type,
        "source_id": meta.source_id, "properties": {"started_at_utc": start_wall,
        "channel_count": meta.channel_count, "channel_labels": meta.channel_labels,
        "nominal_sampling_rate_hz": meta.nominal_srate, "channel_format": meta.channel_format,
        "signal_units": "unknown", "device": ("simulated LSL test publisher; not Mitsar hardware"
            if meta.source_id.startswith("SIMULATED_") else "recorded CSV replay; not live hardware"
            if meta.source_id.startswith("RECORDED_CSV_") else "Mitsar SmartBCI x24 (reported; LSL source unverified)"),
        "eeg_csv_path": str(recorder.csv_path)}}
    if neo4j:
        neo4j.submit_session(session_record)
    base_metadata = {"session_id": session_id, "stream_name": meta.name, "stream_type": meta.stream_type,
        "source_id": meta.source_id, "stream_uid": meta.stream_id, "channel_count": meta.channel_count,
        "channel_labels_in_stream_order": meta.channel_labels, "nominal_sampling_rate_hz": meta.nominal_srate,
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
        "prediction_enabled": infer, "unit_warning": "Physical scale and raw/processed status are unverified."}
    started = time.monotonic()
    print(f"Connected to {meta.name!r}; recording to {recorder.csv_path}. Ctrl+C to stop safely.")
    session_logger.info("Connected to stream=%s source_id=%s channels=%d nominal_rate=%s",
                        meta.name, meta.source_id, meta.channel_count, meta.nominal_srate)
    if model:
        print("WARNING: exploratory condition predictions only; model has no external validation and signal units/raw status remain unverified.")
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
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
            if raw_chunk_collector:
                raw_chunk_collector.add(chunk, stamp_values, recorder.sample_count - len(chunk))
            if event_inlet is not None:
                event_samples, event_timestamps = event_inlet.pull_chunk(timeout=0.0, max_samples=256)
                if event_samples:
                    recorder.write_events(event_samples, event_timestamps)
                    if neo4j:
                        for event_sample, event_timestamp in zip(event_samples, event_timestamps):
                            neo4j.submit_event({"session_id": session_id,
                                "event_id": f"{session_id}:{event_timestamp}:{event_sample}",
                                "timestamp": float(event_timestamp), "values": [str(v) for v in event_sample]})
            if not infer:
                continue
            eeg = np.asarray(chunk, dtype=float)[:, model_indices]
            windows = buffer.push(eeg, stamp_values)
            for signal_window, t0, t1, start_sample in windows:
                rate_result = estimate_rate(timestamp_history, meta.nominal_srate, expected, args.rate_tolerance)
                if not rate_result["compatible"]:
                    raise RuntimeError(f"timestamp-derived rate is not compatible with model: {rate_result}")
                window_count += 1
                result = model.predict_window(signal_window, t0, t1, window_count)
                result["window_start_sample"] = start_sample
                recorder.write_prediction(result)
                print(f"window {window_count}: {result['prediction_text']} "
                      f"(baseline={result.get('baseline_probability', float('nan')):.3f}, "
                      f"stress={result.get('stress_probability', float('nan')):.3f})")
                session_logger.info("window=%d prediction=%s baseline_probability=%s stress_probability=%s",
                    window_count, result["predicted_condition"], result.get("baseline_probability"),
                    result.get("stress_probability"))
                if neo4j:
                    neo4j.submit_prediction({"session_id": session_id,
                        "prediction_id": f"{session_id}:{window_count}", "properties": result})
    except KeyboardInterrupt:
        print("Stop requested; flushing local recording.")
    finally:
        end_wall = datetime.now(timezone.utc).isoformat()
        if raw_chunk_collector:
            raw_chunk_collector.flush(recorder.sample_count - len(raw_chunk_collector.samples))
        session_logger.info("Stopping session; samples=%d windows=%d", recorder.sample_count, window_count)
        rate_result = estimate_rate(timestamp_history, meta.nominal_srate, expected or 0.0, args.rate_tolerance)
        final_metadata = {**base_metadata, "recording_end_utc": end_wall,
            "sample_count": recorder.sample_count,
            "timestamp_count": timestamp_count,
            "recording_duration_seconds": (last_timestamp - first_timestamp) if first_timestamp is not None and last_timestamp is not None else 0,
            "timestamp_diagnostics_sample_count": len(timestamp_history),
            "empty_chunk_count": empty_chunk_count, "detected_chunk_interruption_count": interruption_count,
            "event_sample_count": recorder.event_count,
            "raw_eeg_sample_limit_for_db": db_raw_sample_limit,
            "raw_eeg_samples_queued_for_db": raw_chunk_collector.samples_queued if raw_chunk_collector else 0,
            "raw_eeg_chunks_queued_for_db": raw_chunk_collector.chunk_index if raw_chunk_collector else 0,
            "sampling_rate_validation": rate_result, "window_count": window_count,
            "model_algorithm": model.metadata.get("model_algorithm") if model else None,
            "model_cv_accuracy": model.metadata.get("selected_model_cv_metrics", {}).get("accuracy") if model else None,
            "model_cv_macro_f1": model.metadata.get("selected_model_cv_metrics", {}).get("f1_macro") if model else None,
            "prediction_interpretation": "experimental condition, not clinical diagnosis" if model else None,
            "event_stream_status": "recorded separately" if event_inlet else "not selected; inspect discover output and pass --event-stream-index if present",
            "events_csv_path": str(recorder.events_path) if event_inlet else None,
            "log_path": str(log_path)}
        recorder.write_metadata(final_metadata)
        recorder.close()
        try:
            inlet.close_stream()
            if event_inlet:
                event_inlet.close_stream()
        except Exception:
            logging.exception("Error while closing LSL inlet")
        if neo4j:
            session_record["properties"].update({"ended_at_utc": end_wall,
                "sample_count": recorder.sample_count, "window_count": window_count,
                "raw_eeg_sample_count": raw_chunk_collector.samples_queued if raw_chunk_collector else 0,
                "raw_eeg_chunk_count": raw_chunk_collector.chunk_index if raw_chunk_collector else 0,
                "observed_sampling_rate_hz": rate_result["observed_rate_hz"],
                "metadata_json_path": str(recorder.metadata_path),
                "predictions_csv_path": str(recorder.predictions_path),
                "events_csv_path": str(recorder.events_path) if event_inlet else None,
                "log_path": str(log_path)})
            try:
                neo4j.submit_session(session_record)
                neo4j.close()
            except Exception:
                logging.exception("Neo4j finalization failed; local CSV and JSON have been flushed")
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
    inspect = sub.add_parser("inspect", help="inspect metadata for a selected stream")
    inspect.add_argument("--stream-index", type=int, required=True)
    inspect.add_argument("--resolve-wait", type=float, default=2.0)
    db_list = sub.add_parser("db-list", help="retrieve recent session records from Neo4j")
    db_list.add_argument("--limit", type=int, default=20)
    db_show = sub.add_parser("db-show", help="retrieve one session and its predictions/events from Neo4j")
    db_show.add_argument("--session-id", required=True)
    db_samples = sub.add_parser("db-samples", help="retrieve at most a requested duration of stored raw EEG chunks")
    db_samples.add_argument("--session-id", required=True)
    db_samples.add_argument("--duration-seconds", type=float, default=60.0)
    db_samples.add_argument("--output", type=Path)
    for name, help_text in (("record", "record original LSL samples locally"),
                            ("predict", "record and run trained-model window predictions"),
                            ("live", "stream predictions continuously")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--stream-index", type=int, required=True, help="explicit index from discover output")
        cmd.add_argument("--event-stream-index", type=int, help="optional separate marker/event stream index")
        cmd.add_argument("--duration", type=float, help="seconds; omit to run until Ctrl+C")
        cmd.add_argument("--resolve-wait", type=float, default=2.0)
        cmd.add_argument("--connect-timeout", type=float, default=10.0)
        cmd.add_argument("--disconnect-timeout", type=float, default=10.0,
                         help="stop and flush if no EEG samples arrive within this many seconds")
        cmd.add_argument("--rate-tolerance", type=float, default=0.02)
        cmd.add_argument("--db-raw-seconds", type=float, default=0.0,
                         help="store only this initial duration of raw stream samples in Neo4j chunks (requires --neo4j)")
        cmd.add_argument("--output", type=Path, default=ROOT / "data" / "live")
        cmd.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        cmd.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        cmd.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
        cmd.add_argument("--neo4j", action="store_true", help="store session and predictions in Neo4j; requires NEO4J_* environment")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            print_streams(resolve_streams(args.resolve_wait)); return 0
        if args.command == "inspect":
            streams = resolve_streams(args.resolve_wait); print_streams(streams)
            selected = choose_stream(streams, args.stream_index)
            inlet = connect(selected, 10.0)
            try:
                print(json.dumps(extract_stream_metadata(inlet.info()).__dict__, indent=2))
            finally:
                inlet.close_stream()
            return 0
        if args.command in {"db-list", "db-show", "db-samples"}:
            from src.storage.neo4j_store import Neo4jReader
            reader = Neo4jReader()
            try:
                if args.command == "db-list":
                    result = reader.list_sessions(args.limit)
                elif args.command == "db-show":
                    result = reader.get_session(args.session_id)
                else:
                    if args.duration_seconds <= 0:
                        raise ValueError("duration-seconds must be positive")
                    session = reader.get_session(args.session_id)
                    if session is None:
                        result = None
                    else:
                        stored_rate = session["session"].get("observed_sampling_rate_hz") or 500.0
                        max_samples = int(round(args.duration_seconds * float(stored_rate)))
                        labels, samples, timestamps, actual_rate = reader.export_sample_chunks(args.session_id, max_samples)
                        output_path = args.output or ROOT / "data" / "live" / "retrieved" / f"{args.session_id}_up_to_{args.duration_seconds:g}s.csv"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        import csv
                        with output_path.open("w", encoding="utf-8", newline="") as stream:
                            writer = csv.writer(stream, lineterminator="\n")
                            writer.writerow(["sample_index", "lsl_timestamp", *labels])
                            for index, (row, timestamp) in enumerate(zip(samples, timestamps)):
                                writer.writerow([index, timestamp, *row])
                        result = {"session_id": args.session_id, "output_csv": str(output_path),
                            "samples_retrieved": len(samples), "duration_seconds_at_recorded_rate": len(samples) / actual_rate,
                            "sampling_rate_hz": actual_rate, "channel_labels": labels}
            finally:
                reader.close()
            if result is None:
                print("No matching session found.")
                return 1
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        return _run(args, infer=args.command in {"predict", "live"})
    except Exception as exc:
        parser.exit(2, f"Acquisition stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
