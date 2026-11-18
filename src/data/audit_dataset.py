#!/usr/bin/env python3
"""Stage 1 audit for semicolon-delimited EEG recordings (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_EEG = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
EVENT_RE = re.compile(r"\b(baseline[_ ]start|baseline[_ ]end|stress[_ ]start|stress[_ ]end)\b", re.I)
TIME_RE = re.compile(r"(?<!\d)(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)")
ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\b")
MISSING = {"", "na", "n/a", "nan", "null", "none", "missing"}


def number(value: str) -> float | None:
    try:
        result = float(value.strip())
    except (ValueError, AttributeError):
        return None
    return result if math.isfinite(result) else None


def parse_timestamp(value: str) -> datetime | None:
    match = ISO_RE.search(value)
    if match:
        raw = match.group(1)
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                pass
    match = TIME_RE.search(value)
    if match:
        raw = match.group(1)
        for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                pass
    return None


def clean_header(value: str, index: int) -> str:
    value = value.strip().lstrip("\ufeff")
    return value if value else f"__unnamed_{index + 1}"


def audit_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        header_line = stream.readline()
        # Most exports are semicolon-delimited, but detect a different
        # delimiter when the header clearly contains more of it.
        delimiter = ";" if header_line.count(";") >= header_line.count(",") else ","
        stream.seek(0)
        reader = csv.reader(stream, delimiter=delimiter, strict=False)
        try:
            raw_header = next(reader)
        except StopIteration:
            return {"file": path.name, "participant_id": path.stem, "error": "empty file"}

        headers = [clean_header(v, i) for i, v in enumerate(raw_header)]
        normalized = {h.lower(): h for h in headers}
        eeg_columns = [normalized[ch.lower()] for ch in EXPECTED_EEG if ch.lower() in normalized]
        absent_eeg = [ch for ch in EXPECTED_EEG if ch.lower() not in normalized]
        field_stats: dict[str, dict[str, Any]] = {
            h: {"missing": 0, "numeric": 0, "non_numeric": 0, "non_finite": 0,
                "min": None, "max": None, "first_values": []}
            for h in headers
        }
        event_counts: Counter[str] = Counter()
        event_locations: dict[str, Counter[str]] = defaultdict(Counter)
        other_text_values: dict[str, Counter[str]] = defaultdict(Counter)
        time_counts: Counter[str] = Counter()
        timestamp_seconds: dict[str, list[float]] = defaultdict(list)
        timestamp_rows: Counter[str] = Counter()
        row_count = malformed_width = short_rows = long_rows = 0
        width_counts: Counter[int] = Counter()

        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            row_count += 1
            width = len(row)
            width_counts[width] += 1
            if width != len(headers):
                malformed_width += 1
                if width < len(headers):
                    short_rows += 1
                else:
                    long_rows += 1
            # Map by header position and retain all overflow cells explicitly.
            for i, h in enumerate(headers):
                value = row[i].strip() if i < width else ""
                stats = field_stats[h]
                if value.lower() in MISSING:
                    stats["missing"] += 1
                else:
                    val = number(value)
                    if val is None:
                        if value.lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                            stats["non_finite"] += 1
                        else:
                            stats["non_numeric"] += 1
                            if len(stats["first_values"]) < 5 and value not in stats["first_values"]:
                                stats["first_values"].append(value[:100])
                    else:
                        stats["numeric"] += 1
                        stats["min"] = val if stats["min"] is None else min(stats["min"], val)
                        stats["max"] = val if stats["max"] is None else max(stats["max"], val)

                if h not in eeg_columns and value:
                    for match in EVENT_RE.finditer(value):
                        label = re.sub(r"[ _]+", "_", match.group(1)).lower()
                        event_counts[label] += 1
                        event_locations[label][h] += 1
                        ts = parse_timestamp(value)
                        if ts:
                            time_counts[h] += 1
                            # Time-of-day only; wraparound means this is not a duration.
                            timestamp_seconds[h].append(ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6)
                    if number(value) is None and value.lower() not in MISSING:
                        if not EVENT_RE.search(value) and len(other_text_values[h]) < 30:
                            other_text_values[h][value[:100]] += 1
                    ts = parse_timestamp(value)
                    if ts and not EVENT_RE.search(value):
                        time_counts[h] += 1
                        timestamp_rows[h] += 1
                        timestamp_seconds[h].append(ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6)

            # Include event/timestamp values in overflow fields beyond the header.
            for j in range(len(headers), width):
                value = row[j].strip()
                h = f"__overflow_{j - len(headers) + 1}"
                if not value or value.lower() in MISSING:
                    continue
                for match in EVENT_RE.finditer(value):
                    label = re.sub(r"[ _]+", "_", match.group(1)).lower()
                    event_counts[label] += 1
                    event_locations[label][h] += 1
                if parse_timestamp(value):
                    time_counts[h] += 1
                    timestamp_rows[h] += 1
                    ts = parse_timestamp(value)
                    timestamp_seconds[h].append(ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6)
                elif number(value) is None and len(other_text_values[h]) < 30:
                    other_text_values[h][value[:100]] += 1

    # The event interval summary is based on explicit marker text only.
    interval_completeness = {}
    for condition in ("baseline", "stress"):
        start = event_counts.get(f"{condition}_start", 0)
        end = event_counts.get(f"{condition}_end", 0)
        interval_completeness[condition] = {"start_markers": start, "end_markers": end,
                                            "has_complete_marker_pair": start > 0 and end > 0}

    channel_summary = {}
    for channel in eeg_columns:
        st = field_stats[channel]
        valid = st["numeric"]
        channel_summary[channel] = {
            "numeric_values": valid,
            "missing_values": st["missing"],
            "non_numeric_values": st["non_numeric"],
            "non_finite_values": st["non_finite"],
            "min": st["min"], "max": st["max"],
            "constant_or_flatline_candidate": valid > 0 and st["min"] == st["max"],
            "non_numeric_examples": st["first_values"],
        }

    column_types = {}
    for name, st in field_stats.items():
        seen_numeric = st["numeric"] > 0
        seen_text = st["non_numeric"] > 0
        if seen_numeric and seen_text:
            inferred = "mixed numeric and text"
        elif seen_numeric:
            inferred = "numeric"
        elif seen_text:
            inferred = "text"
        else:
            inferred = "empty"
        if st["non_finite"]:
            inferred += " (contains non-finite values)"
        column_types[name] = {
            "inferred_type": inferred,
            "numeric_values": st["numeric"],
            "text_values": st["non_numeric"],
            "missing_values": st["missing"],
            "non_finite_values": st["non_finite"],
        }

    time_summary = {}
    for field, vals in timestamp_seconds.items():
        # A duration is reported only if timestamp values occur on nearly every row.
        coverage = timestamp_rows[field] / row_count if row_count else 0
        duration = None
        if coverage >= 0.9 and vals:
            duration = max(vals) - min(vals)
        time_summary[field] = {"timestamp_values": len(vals), "row_coverage": coverage,
                               "span_seconds_if_continuous": duration,
                               "duration_established": duration is not None}

    auxiliary = [h for h in headers if h not in eeg_columns]
    return {
        "file": path.name,
        "participant_id": path.stem,
        "rows": row_count,
        "columns": len(headers),
        "delimiter": delimiter,
        "headers": headers,
        "eeg_channels_found": eeg_columns,
        "missing_expected_eeg_channels": absent_eeg,
        "auxiliary_fields": auxiliary,
        "column_data_types": column_types,
        "row_width_counts": {str(k): v for k, v in sorted(width_counts.items())},
        "rows_with_width_mismatch": malformed_width,
        "short_rows": short_rows,
        "long_rows": long_rows,
        "event_marker_counts": dict(sorted(event_counts.items())),
        "event_marker_fields": {k: dict(v) for k, v in event_locations.items()},
        "condition_interval_markers": interval_completeness,
        "timestamp_fields": time_summary,
        "duration_seconds": None,
        "duration_status": "not established from per-row timestamps",
        "eeg_signal_quality": channel_summary,
        "auxiliary_field_data_quality": {
            h: {"missing": field_stats[h]["missing"],
                "non_numeric": field_stats[h]["non_numeric"],
                "non_finite": field_stats[h]["non_finite"],
                "non_numeric_examples": field_stats[h]["first_values"]}
            for h in auxiliary
        },
        "non_numeric_auxiliary_values": {h: dict(c) for h, c in other_text_values.items()},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# EEG dataset audit", "", f"Generated: {report['generated_at']}",
             f"Files found: **{report['files_found']}**", "",
             "## Dataset-level findings", "",
             f"- Participant CSV files: {report['files_found']}",
             f"- Files successfully audited: {report['files_audited']}",
             f"- Files with explicit baseline start and end markers: {report['baseline_complete_files']}",
             f"- Files with explicit stress start and end markers: {report['stress_complete_files']}",
             f"- Sampling rate: **not established**", f"- Physical signal units: **not established**", "",
             "## Per-file summary", "",
             "| File | Rows | EEG channels found | Missing expected channels | Width-mismatch rows | Marker counts | Duration |",
             "|---|---:|---:|---|---:|---|---|"]
    for item in report["files"]:
        if "error" in item:
            lines.append(f"| {item['file']} | — | — | — | — | — | Error: {item['error']} |")
            continue
        markers = ", ".join(f"{k}={v}" for k, v in item["event_marker_counts"].items()) or "none detected"
        missing = ", ".join(item["missing_expected_eeg_channels"]) or "none"
        lines.append(f"| {item['file']} | {item['rows']} | {len(item['eeg_channels_found'])}/19 | {missing} | {item['rows_with_width_mismatch']} | {markers} | not established |")
    lines += ["", "## Detailed findings", ""]
    for item in report["files"]:
        if "error" in item:
            continue
        lines += [f"### {item['file']}", "",
                  f"- EEG fields: {', '.join(item['eeg_channels_found']) or 'none detected'}",
                  f"- Auxiliary fields: {', '.join(item['auxiliary_fields']) or 'none detected'}",
                  f"- Row widths (data fields per row): {item['row_width_counts']}",
                  f"- Event markers: {item['event_marker_counts'] or 'none detected'}",
                  f"- Marker locations: {item['event_marker_fields'] or 'none detected'}",
                  f"- Baseline markers: {item['condition_interval_markers']['baseline']}; stress markers: {item['condition_interval_markers']['stress']}",
                  f"- Timestamp findings: {item['timestamp_fields'] or 'none detected'}",
                  f"- Signal anomalies (missing/non-numeric/non-finite/constant): "
                  f"{sum(1 for x in item['eeg_signal_quality'].values() if x['missing_values'] or x['non_numeric_values'] or x['non_finite_values'] or x['constant_or_flatline_candidate'])} EEG channel(s) flagged.", ""]
    lines += ["## Suitability for the next stage", "",
              "This audit does not establish sampling frequency or physical units. It also does not establish that event labels are clinically validated stress measurements. Files without explicit, paired baseline and stress interval markers cannot be labeled from this audit. Row-width inconsistencies and annotation placement must be resolved before building a loader or extracting windows.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/audit"))
    args = parser.parse_args()
    files = sorted(args.data_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not files:
        parser.error(f"No CSV files found in {args.data_dir}")
    results = []
    for path in files:
        print(f"Auditing {path.name}...", flush=True)
        try:
            results.append(audit_file(path))
        except Exception as exc:
            results.append({"file": path.name, "participant_id": path.stem,
                            "error": f"{type(exc).__name__}: {exc}"})
    audited = [r for r in results if "error" not in r]
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_directory": str(args.data_dir),
        "files_found": len(files),
        "files_audited": len(audited),
        "baseline_complete_files": sum(r["condition_interval_markers"]["baseline"]["has_complete_marker_pair"] for r in audited),
        "stress_complete_files": sum(r["condition_interval_markers"]["stress"]["has_complete_marker_pair"] for r in audited),
        "sampling_rate_hz": None,
        "physical_units": None,
        "files": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "dataset_audit.json"
    md_path = args.out_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Audited {len(audited)}/{len(files)} files. Reports: {json_path}, {md_path}")
    return 0 if len(audited) == len(files) else 1

if __name__ == "__main__":
    sys.exit(main())
