"""Read locally archived EEG session metadata and prediction CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "data" / "live" / "combined-app"


def _metadata_files(root: Path = DEFAULT_ARCHIVE) -> list[Path]:
    folder = root / "metadata"
    return sorted(folder.glob("stream_metadata_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.exists() else []


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read session metadata {path.name}: {exc}") from exc
    value["_metadata_path"] = str(path)
    return value


def list_sessions(limit: int = 50, root: Path = DEFAULT_ARCHIVE) -> list[dict[str, Any]]:
    sessions = []
    for path in _metadata_files(root):
        try:
            record = _read_metadata(path)
        except ValueError:
            continue
        summary = record.get("summary") or {}
        sample_count = int(record.get("sample_count", summary.get("sample_count", 0)) or 0)
        if sample_count <= 0:
            continue
        sessions.append({
            "session_id": record.get("session_id"),
            "participant_id": record.get("participant_id"),
            "participant_name": record.get("participant_name"),
            "started_at_utc": record.get("recording_start_utc"),
            "ended_at_utc": record.get("recording_end_utc"),
            "stream_name": record.get("stream_name"),
            "device": record.get("device", "unknown"),
            "channel_count": record.get("channel_count"),
            "channel_labels": record.get("channel_labels_in_stream_order", []),
            "channel_labels_used": record.get("channel_labels_used", []),
            "nominal_sampling_rate_hz": record.get("nominal_sampling_rate_hz"),
            "observed_sampling_rate_hz": (record.get("sampling_rate_validation") or {}).get("observed_rate_hz"),
            "sample_count": sample_count,
            "window_count": int(record.get("window_count", summary.get("window_count", 0)) or 0),
            "prediction_count_already_stored": int(record.get("window_count", summary.get("window_count", 0)) or 0),
            "prediction_enabled": bool(record.get("prediction_enabled", False)),
            "inference_block_reason": record.get("inference_block_reason"),
            "metadata_json_path": str(path),
        })
    return sessions[:max(1, min(int(limit), 500))]


def session_report(session_id: str, root: Path = DEFAULT_ARCHIVE) -> dict[str, Any]:
    for path in _metadata_files(root):
        record = _read_metadata(path)
        if record.get("session_id") != session_id:
            continue
        prediction_path_value = record.get("predictions_csv_path")
        predictions = []
        if prediction_path_value:
            prediction_path = Path(prediction_path_value)
            if not prediction_path.is_absolute():
                prediction_path = root / prediction_path
            elif not prediction_path.exists():
                # Archives remain readable if the whole EEGRT folder was moved.
                prediction_path = root / "predictions" / prediction_path.name
            if prediction_path.exists():
                with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        for key in ("window_index", "window_start_timestamp", "window_end_timestamp",
                                    "baseline_probability", "stress_probability"):
                            if row.get(key) not in (None, ""):
                                try:
                                    row[key] = float(row[key]) if "probability" in key or "timestamp" in key else int(row[key])
                                except ValueError:
                                    pass
                        try:
                            row["elapsed_seconds"] = float(row.get("elapsed_seconds") or 0)
                        except ValueError:
                            row["elapsed_seconds"] = 0.0
                        predictions.append(row)
        metadata = {key: value for key, value in record.items() if not key.startswith("_")}
        session = {
            "session_id": metadata.get("session_id"),
            "participant_id": metadata.get("participant_id"),
            "participant_name": metadata.get("participant_name"),
            "started_at_utc": metadata.get("recording_start_utc"),
            "ended_at_utc": metadata.get("recording_end_utc"),
            "stream_name": metadata.get("stream_name"),
            "device": metadata.get("device", "unknown"),
            "channel_count": metadata.get("channel_count"),
            "channel_labels": metadata.get("channel_labels_in_stream_order", []),
            "channel_labels_in_stream_order": metadata.get("channel_labels_in_stream_order", []),
            "channel_labels_used": metadata.get("channel_labels_used", []),
            "nominal_sampling_rate_hz": metadata.get("nominal_sampling_rate_hz"),
            "observed_sampling_rate_hz": (metadata.get("sampling_rate_validation") or {}).get("observed_rate_hz"),
            "sample_count": metadata.get("sample_count", 0),
            "window_count": metadata.get("window_count", len(predictions)),
            "prediction_enabled": metadata.get("prediction_enabled", False),
            "inference_block_reason": metadata.get("inference_block_reason"),
            "model_algorithm": metadata.get("model_algorithm"),
            "model_validation_accuracy": metadata.get("model_cv_accuracy"),
            "model_validation_macro_f1": metadata.get("model_cv_macro_f1"),
            "average_stress_score": (metadata.get("summary") or {}).get("average_stress_score"),
            "stress_window_count": (metadata.get("summary") or {}).get("stress_window_count", 0),
            "baseline_window_count": (metadata.get("summary") or {}).get("baseline_window_count", 0),
            "majority_window_condition": (metadata.get("summary") or {}).get("majority_window_condition"),
        }
        return {"session": session, "summary": metadata.get("summary", {}),
                "predictions": predictions, "model_algorithm": metadata.get("model_algorithm")}
    raise ValueError(f"Local session not found: {session_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sessions = commands.add_parser("sessions", help="list locally saved sessions")
    sessions.add_argument("--limit", type=int, default=50)
    report = commands.add_parser("session-report", help="read a local session and its saved predictions")
    report.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    try:
        payload = ({"ok": True, "sessions": list_sessions(args.limit)} if args.command == "sessions"
                   else {"ok": True, **session_report(args.session_id)})
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
