"""Read participant/session metadata and saved model results from Neo4j Aura.

Live EEG acquisition and inference run directly in the combined Electron app
through ``src.acquisition.cli``. This service only supports the session archive
and report download; it never reads or stores raw EEG samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _reader_type():
    from src.storage.neo4j_store import Neo4jReader
    return Neo4jReader


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return completed sessions and prediction counts from Aura."""
    Reader = _reader_type()
    reader = Reader()
    try:
        rows = reader.list_sessions(limit)
    finally:
        reader.close()
    sessions = []
    for row in rows:
        session = row["session"]
        sample_count = int(session.get("sample_count", 0) or 0)
        if sample_count <= 0:
            continue
        sessions.append({
            "session_id": session.get("session_id"),
            "participant_id": session.get("participant_id"),
            "participant_name": session.get("participant_name"),
            "started_at_utc": session.get("started_at_utc"),
            "ended_at_utc": session.get("ended_at_utc"),
            "stream_name": session.get("stream_name"),
            "device": session.get("device", "unknown"),
            "channel_count": session.get("channel_count"),
            "nominal_sampling_rate_hz": session.get("nominal_sampling_rate_hz"),
            "observed_sampling_rate_hz": session.get("observed_sampling_rate_hz"),
            "sample_count": sample_count,
            "window_count": int(session.get("window_count", 0) or 0),
            "prediction_count_already_stored": int(row.get("prediction_count", 0) or 0),
            "prediction_enabled": bool(session.get("prediction_enabled", False)),
            "inference_block_reason": session.get("inference_block_reason"),
        })
    return sessions


def session_report(session_id: str) -> dict[str, Any]:
    """Return the saved participant/session properties and window predictions."""
    Reader = _reader_type()
    reader = Reader()
    try:
        result = reader.get_session_report(session_id)
    finally:
        reader.close()
    if result is None:
        raise ValueError(f"session not found: {session_id}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sessions = commands.add_parser("sessions", help="list completed sessions and prediction results")
    sessions.add_argument("--limit", type=int, default=50)
    report = commands.add_parser("session-report", help="retrieve one saved session and its predictions")
    report.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "sessions":
            payload = {"ok": True, "sessions": list_sessions(args.limit)}
        else:
            payload = {"ok": True, **session_report(args.session_id)}
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
