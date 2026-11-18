#!/usr/bin/env python3
"""Create condition-contained window index rows; never infer a sample rate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

WINDOW_FIELDS = [
    "participant_id", "condition", "label", "source_filename", "normalized_filename",
    "start_sample_index", "end_sample_index_exclusive",
    "condition_start_marker_timestamp", "condition_end_marker_timestamp",
]


def window_indices(interval_start: int, interval_end: int, sample_rate: float,
                   window_seconds: float, overlap_fraction: float) -> list[tuple[int, int]]:
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sampling frequency and window duration must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("window overlap fraction must be in [0, 1)")
    length = int(round(sample_rate * window_seconds))
    if length < 1:
        raise ValueError("window duration resolves to fewer than one sample")
    step = max(1, int(round(length * (1 - overlap_fraction))))
    return [(start, start + length) for start in range(interval_start, interval_end - length + 1, step)]


def config_sampling_rate(config: dict[str, Any]) -> float:
    fs = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or fs is None:
        raise ValueError("sampling frequency is unset or unverified; cannot convert window seconds to samples")
    fs = float(fs)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("verified sampling frequency must be a finite positive number")
    return fs


def build_window_rows(marker_report: dict[str, Any], manifest: dict[str, dict[str, str]],
                      config: dict[str, Any]) -> list[dict[str, Any]]:
    fs = config_sampling_rate(config)
    overlap = float(config["window_overlap_fraction"])
    duration = float(config["window_duration_seconds"])
    label_map = config["label_map"]
    rows: list[dict[str, Any]] = []
    for participant in marker_report["participants"]:
        if not participant["valid_for_windowing"]:
            continue
        mrow = manifest.get(participant["participant_id"], {})
        for condition, start_marker, end_marker in (
            ("baseline", "Baseline_start", "Baseline_end"),
            ("stress", "stress_start", "stress_end"),
        ):
            interval = participant["intervals"][condition]
            start_hits, end_hits = participant["markers"][start_marker], participant["markers"][end_marker]
            for start, end in window_indices(interval["start_sample_index"], interval["end_sample_index"], fs, duration, overlap):
                rows.append({
                    "participant_id": participant["participant_id"],
                    "condition": condition,
                    "label": label_map[condition],
                    "source_filename": participant["source_filename"],
                    "normalized_filename": mrow.get("normalized_filename", ""),
                    "start_sample_index": start,
                    "end_sample_index_exclusive": end,
                    "condition_start_marker_timestamp": start_hits[0].get("timestamp") or "",
                    "condition_end_marker_timestamp": end_hits[0].get("timestamp") or "",
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eeg_config.json"))
    parser.add_argument("--marker-report", type=Path, default=Path("data/processed/marker_validation.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads(args.marker_report.read_text(encoding="utf-8"))
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {row["participant_id"]: row for row in csv.DictReader(stream)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "window_index.csv"
    result_path = args.output_dir / "window_preparation_report.md"
    try:
        rows = build_window_rows(report, manifest, config)
        reason = ""
    except (ValueError, KeyError) as exc:
        rows, reason = [], str(exc)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=WINDOW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    valid_count = sum(bool(p["valid_for_windowing"]) for p in report["participants"])
    result_path.write_text(
        "# Window preparation report\n\n"
        f"- Participants with unambiguous, ordered, non-overlapping markers: **{valid_count}**\n"
        f"- Window index rows created: **{len(rows)}**\n"
        f"- Blocker: **{reason or 'none'}**\n\n"
        "The window index points to contiguous samples in normalized recordings; it does not copy or transform samples. "
        "Incomplete trailing fragments are omitted. Windows never cross condition interval boundaries. No filtering or feature extraction is performed.\n",
        encoding="utf-8",
    )
    print(f"Window index: {len(rows)} rows. {reason or 'Created successfully.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
