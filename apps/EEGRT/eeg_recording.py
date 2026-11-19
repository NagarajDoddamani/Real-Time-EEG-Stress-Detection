"""Incremental local CSV and JSON recording helpers."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalRecording:
    def __init__(self, root: Path, session_id: str, labels: list[str]):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.root = root
        self.csv_path = root / "recordings" / f"eeg_samples_{stamp}.csv"
        self.metadata_path = root / "metadata" / f"stream_metadata_{stamp}.json"
        self.predictions_path = root / "predictions" / f"predictions_{stamp}.csv"
        self.events_path = root / "events" / f"events_{stamp}.csv"
        for path in (self.csv_path, self.metadata_path, self.predictions_path, self.events_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id, self.labels = session_id, labels
        self.sample_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self.sample_writer = csv.writer(self.sample_file, lineterminator="\n")
        self.sample_writer.writerow(["sample_index", "lsl_timestamp", *labels])
        self.pred_file = self.predictions_path.open("w", encoding="utf-8", newline="")
        self.pred_writer = csv.DictWriter(self.pred_file, fieldnames=["window_index", "window_start_timestamp",
            "window_end_timestamp", "elapsed_seconds", "prediction", "predicted_condition", "baseline_probability",
            "stress_probability", "model_algorithm", "model_version"], lineterminator="\n")
        self.pred_writer.writeheader()
        self.event_file = self.events_path.open("w", encoding="utf-8", newline="")
        self.event_writer = csv.writer(self.event_file, lineterminator="\n")
        self.event_writer.writerow(["lsl_timestamp", "event_values"])
        self.sample_count = 0
        self.event_count = 0

    def write_samples(self, samples: list[list[float]], timestamps: list[float]) -> None:
        for row, timestamp in zip(samples, timestamps):
            self.sample_writer.writerow([self.sample_count, timestamp, *row])
            self.sample_count += 1
        self.sample_file.flush()

    def write_prediction(self, result: dict[str, Any]) -> None:
        self.pred_writer.writerow({key: result.get(key) for key in self.pred_writer.fieldnames})
        self.pred_file.flush()

    def write_events(self, samples: list[list[Any]], timestamps: list[float]) -> None:
        if len(samples) != len(timestamps):
            raise ValueError("event sample/timestamp count mismatch")
        for values, timestamp in zip(samples, timestamps):
            self.event_writer.writerow([timestamp, json.dumps([str(value) for value in values], ensure_ascii=False)])
            self.event_count += 1
        self.event_file.flush()

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    def close(self) -> None:
        self.sample_file.flush(); self.sample_file.close()
        self.pred_file.flush(); self.pred_file.close()
        self.event_file.flush(); self.event_file.close()

