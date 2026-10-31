#!/usr/bin/env python3
"""Validate raw EEG CSVs and prepare normalized, eligible-only recordings.

This stage only selects the 19 EEG columns and validates explicit condition
markers. It does not filter, resample, transform units, extract features, or
train a model. Python standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
MARKERS = ("Baseline_start", "Baseline_end", "stress_start", "stress_end")
MARKER_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, MARKERS)) + r")(?![A-Za-z0-9])", re.I)
TIME_RE = re.compile(r"(?<!\d)(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)(?!\d)")
ISO_TIME_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\b")
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "missing"}


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value.strip()))
    except (TypeError, ValueError, OverflowError):
        return False


def timestamp_from_text(value: str) -> str | None:
    """Return a timestamp exactly as represented in text; never invent one."""
    match = ISO_TIME_RE.search(value) or TIME_RE.search(value)
    return match.group(1) if match else None


def canonical_marker(value: str) -> str:
    lowered = value.lower()
    return next(m for m in MARKERS if m.lower() == lowered)


def choose_delimiter(header_line: str) -> str:
    semicolons, commas = header_line.count(";"), header_line.count(",")
    if semicolons == commas:
        raise ValueError("could not determine delimiter from header (comma/semicolon count tied)")
    return ";" if semicolons > commas else ","


def safe_header(raw: str, index: int) -> str:
    raw = raw.strip().lstrip("\ufeff")
    return raw if raw else f"__unnamed_{index + 1}"


def _empty_marker_record() -> dict[str, Any]:
    return {marker: [] for marker in MARKERS}


def inspect_recording(path: Path, staging_dir: Path) -> dict[str, Any]:
    """Inspect one file and retain validated EEG rows only in memory."""
    result: dict[str, Any] = {
        "participant_id": path.stem,
        "source_filename": path.name,
        "delimiter_detected": "",
        "headers": [],
        "total_rows": 0,
        "valid_eeg_rows": 0,
        "eeg_channel_count": 0,
        "format_valid": False,
        "annotations_valid": False,
        "eligible_for_training": False,
        "exclusion_reasons": [],
        "row_width_problems": [],
        "invalid_eeg_rows": [],
        "annotation_markers": _empty_marker_record(),
        "annotation_strings": [],
    }
    reasons: list[str] = result["exclusion_reasons"]
    stage_stream = None
    stage_path: Path | None = None
    annotation_groups: dict[tuple[str, str], dict[str, Any]] = {}

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            header_line = stream.readline()
            if not header_line:
                reasons.append("empty file; no header")
                return result
            delimiter = choose_delimiter(header_line)
            result["delimiter_detected"] = "semicolon" if delimiter == ";" else "comma"
            stream.seek(0)
            reader = csv.reader(stream, delimiter=delimiter, strict=True)
            raw_headers = next(reader)
            headers = [safe_header(name, i) for i, name in enumerate(raw_headers)]
            result["headers"] = headers
            lowered_headers = [name.casefold() for name in headers]
            result["eeg_channel_count"] = sum(channel.casefold() in lowered_headers for channel in EEG_CHANNELS)
            missing_channels = [c for c in EEG_CHANNELS if c.casefold() not in lowered_headers]
            duplicate_headers = sorted({h for h in lowered_headers if lowered_headers.count(h) > 1})
            if missing_channels:
                reasons.append("missing EEG channel(s): " + ", ".join(missing_channels))
            if duplicate_headers:
                reasons.append("duplicate header(s): " + ", ".join(duplicate_headers))

            # Output order is fixed by EEG_CHANNELS, independent of input order.
            channel_positions: list[int] = []
            if not missing_channels and not duplicate_headers:
                channel_positions = [lowered_headers.index(c.casefold()) for c in EEG_CHANNELS]
            eeg_positions = set(channel_positions)
            if channel_positions:
                stage_path = staging_dir / f"{path.stem}.csv.tmp"
                stage_stream = stage_path.open("w", encoding="utf-8", newline="")
                stage_writer = csv.writer(stage_stream, lineterminator="\n")
                stage_writer.writerow(EEG_CHANNELS)
            row_width_counts: Counter[int] = Counter()
            parse_error: str | None = None

            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    parse_error = f"CSV parser error near physical line {reader.line_num}: {exc}"
                    break

                line_number = reader.line_num
                result["total_rows"] += 1
                row_width_counts[len(row)] += 1
                if not row or (len(row) == 1 and not row[0].strip()):
                    result["row_width_problems"].append({
                        "line_number": line_number, "actual_fields": len(row),
                        "header_fields": len(headers), "recoverable": False,
                        "reason": "blank data row contains no recoverable EEG sample",
                    })
                    result["invalid_eeg_rows"].append(line_number)
                    continue

                if len(row) != len(headers):
                    recoverable = bool(channel_positions) and len(row) > max(channel_positions)
                    reason = (
                        "field count differs from header; all 19 EEG positions were recovered by header position"
                        if recoverable else
                        "field count differs from header and one or more required EEG positions are absent"
                    )
                    result["row_width_problems"].append({
                        "line_number": line_number, "actual_fields": len(row),
                        "header_fields": len(headers), "recoverable": recoverable,
                        "reason": reason,
                    })

                if channel_positions:
                    if len(row) <= max(channel_positions):
                        result["invalid_eeg_rows"].append(line_number)
                    else:
                        values = [row[position].strip() for position in channel_positions]
                        bad = [EEG_CHANNELS[i] for i, value in enumerate(values)
                               if value.casefold() in MISSING_TOKENS or not finite_number(value)]
                        if bad:
                            result["invalid_eeg_rows"].append(line_number)
                        else:
                            stage_writer.writerow(values)
                            result["valid_eeg_rows"] += 1

                # Search every non-EEG cell, including unnamed header columns
                # and overflow cells. Preserve raw annotation text verbatim.
                for position, value in enumerate(row):
                    if position in eeg_positions or not value.strip():
                        continue
                    raw_value = value
                    found = list(MARKER_RE.finditer(raw_value))
                    is_missing = raw_value.strip().casefold() in MISSING_TOKENS
                    if found or (not is_missing and not finite_number(raw_value)):
                        field_name = headers[position] if position < len(headers) else f"__overflow_{position-len(headers)+1}"
                        key = (field_name, raw_value)
                        record = annotation_groups.get(key)
                        if record is None:
                            record = {
                                "field_name": field_name,
                                "raw_text": raw_value,
                                "timestamp": timestamp_from_text(raw_value),
                                "markers": [canonical_marker(m.group(1)) for m in found],
                                "source_line_numbers": [],
                            }
                            annotation_groups[key] = record
                        record["source_line_numbers"].append(line_number)

            result["row_width_counts"] = {str(k): v for k, v in sorted(row_width_counts.items())}
            result["annotation_strings"] = list(annotation_groups.values())
            if stage_stream is not None:
                stage_stream.flush()
                stage_stream.close()
                stage_stream = None
            if parse_error:
                reasons.append(parse_error)

            # Unequal rows are acceptable only when the fixed, header-defined
            # EEG positions remain present and are valid on every data row.
            unrecoverable_widths = [p for p in result["row_width_problems"] if not p["recoverable"]]
            if unrecoverable_widths:
                reasons.append(f"{len(unrecoverable_widths)} row(s) have unrecoverable field-width problems")
            if result["invalid_eeg_rows"]:
                reasons.append(f"{len(result['invalid_eeg_rows'])} row(s) contain missing, non-numeric, non-finite, or unrecoverable EEG values")

            result["format_valid"] = bool(
                channel_positions and not parse_error and not unrecoverable_widths
                and not result["invalid_eeg_rows"]
                and result["valid_eeg_rows"] == result["total_rows"]
            )
            result["annotation_markers"] = {
                marker: [
                    item for item in result["annotation_strings"]
                    if marker in item["markers"]
                ] for marker in MARKERS
            }
            result["annotations_valid"] = all(result["annotation_markers"][m] for m in MARKERS)
            if not result["annotations_valid"]:
                missing = [m for m in MARKERS if not result["annotation_markers"][m]]
                reasons.append("missing condition marker(s): " + ", ".join(missing))
            result["eligible_for_training"] = result["format_valid"] and result["annotations_valid"]
    except (OSError, UnicodeError, csv.Error, ValueError, StopIteration) as exc:
        reasons.append(f"could not parse recording: {type(exc).__name__}: {exc}")
    finally:
        if stage_stream is not None:
            stage_stream.close()

    # Keep the streaming copy only for fully eligible recordings.
    if result["eligible_for_training"] and stage_path is not None:
        result["_staged_path"] = str(stage_path)
    elif stage_path is not None:
        stage_path.unlink(missing_ok=True)
    return result


def manifest_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "participant_id": item["participant_id"],
        "source_filename": item["source_filename"],
        "normalized_filename": f"{item['participant_id']}.csv" if item["eligible_for_training"] else "",
        "delimiter_detected": item["delimiter_detected"],
        "eeg_channel_count": item["eeg_channel_count"],
        "total_rows": item["total_rows"],
        "valid_eeg_rows": item["valid_eeg_rows"],
    }
    for marker in MARKERS:
        values = item["annotation_markers"].get(marker, [])
        raw = values[0]["raw_text"] if values else ""
        timestamp = values[0]["timestamp"] if values else None
        row[marker] = timestamp or raw
    row.update({
        "format_valid": item["format_valid"],
        "annotations_valid": item["annotations_valid"],
        "eligible_for_training": item["eligible_for_training"],
        "exclusion_reason": "; ".join(item["exclusion_reasons"]),
    })
    return row


def _write_normalized(item: dict[str, Any], eligible_dir: Path) -> None:
    output = eligible_dir / f"{item['participant_id']}.csv"
    Path(item.pop("_staged_path")).replace(output)


def render_report(items: list[dict[str, Any]], delimiter_counts: Counter[str]) -> str:
    eligible = [i for i in items if i["eligible_for_training"]]
    excluded = [i for i in items if not i["eligible_for_training"]]
    reason_counts: Counter[str] = Counter()
    for item in excluded:
        for reason in item["exclusion_reasons"]:
            reason_counts[reason] += 1
    problem_rows = sum(len(i["row_width_problems"]) for i in items)
    recoverable_rows = sum(sum(p["recoverable"] for p in i["row_width_problems"]) for i in items)
    lines = [
        "# EEG dataset preparation report", "",
        "## Summary", "",
        f"- Total CSV files inspected: **{len(items)}**",
        f"- Eligible recordings: **{len(eligible)}**",
        f"- Excluded recordings: **{len(excluded)}**",
        f"- Delimiter distribution: {dict(sorted(delimiter_counts.items()))}",
        f"- Rows with a field count different from their header: **{problem_rows}** "
        f"({recoverable_rows} had all 19 EEG positions recoverable by header position).",
        "- Sampling frequency: **not established**.",
        "- Physical EEG units: **not established**.", "",
        "## Exclusion reasons", "",
    ]
    if reason_counts:
        lines.extend(f"- {reason} — {count} recording(s)" for reason, count in sorted(reason_counts.items()))
    else:
        lines.append("- None.")
    lines += ["", "## Annotation coverage", "",
              "Markers are searched in every non-EEG field, including unnamed trailing fields. Timestamps below are preserved only when present in the original annotation text.", "",
              "| File | Baseline start | Baseline end | Stress start | Stress end | Valid annotations |",
              "|---|---|---|---|---|---|"]
    for item in items:
        vals = []
        for marker in MARKERS:
            records = item["annotation_markers"].get(marker, [])
            vals.append(", ".join(
                f"line(s) {', '.join(map(str, a['source_line_numbers']))}: {a['timestamp'] or '(timestamp absent)'}"
                for a in records
            ) or "missing")
        lines.append(f"| {item['source_filename']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {item['annotations_valid']} |")
    lines += ["", "## Row-width problems", "",
              "Rows are numbered by physical source-file line. Normalization used the channel positions declared in the header and retained every valid EEG sample. Any unrecoverable row excludes its full recording.", ""]
    any_problems = False
    for item in items:
        if item["row_width_problems"]:
            any_problems = True
            details = "; ".join(
                f"line {p['line_number']} ({p['actual_fields']} fields vs {p['header_fields']}; "
                f"{'recovered' if p['recoverable'] else 'unrecoverable'})"
                for p in item["row_width_problems"]
            )
            lines.append(f"- **{item['source_filename']}**: {details}")
    if not any_problems:
        lines.append("- None detected.")
    lines += ["", "## Final participant lists", "",
              "### Included", ""]
    lines += [f"- {i['participant_id']}" for i in eligible] or ["- None"]
    lines += ["", "### Excluded", ""]
    lines += [f"- **{i['participant_id']}**: {'; '.join(i['exclusion_reasons']) or 'failed validation'}" for i in excluded] or ["- None"]
    lines += ["", "## Scope and readiness", "",
              "Eligible files contain exactly the 19 requested numeric EEG columns, in original sample order, with no invalid EEG values, and explicit occurrences of all four condition markers. Auxiliary fields and annotations are not copied into the normalized EEG CSVs; original annotation text and source line/timestamp metadata are stored in `annotation_metadata.json`. This selection does not validate clinical stress, establish the sampling frequency or physical units, or confirm Mitsar device/streaming details. No filtering or model training was performed.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    files = sorted(args.raw_dir.glob("*.csv"), key=lambda p: p.name.casefold())
    if not files:
        parser.error(f"No CSV files found in {args.raw_dir}")

    results: list[dict[str, Any]] = []
    delimiter_counts: Counter[str] = Counter()
    staging_dir = args.processed_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"Validating {path.name}...", flush=True)
        item = inspect_recording(path, staging_dir)
        results.append(item)
        delimiter_counts[item["delimiter_detected"] or "unknown"] += 1
        status = "ELIGIBLE" if item["eligible_for_training"] else "EXCLUDED"
        print(f"  {status}: {item['total_rows']} rows, {item['valid_eeg_rows']} valid EEG rows")

    eligible_dir = args.processed_dir / "eligible"
    eligible_dir.mkdir(parents=True, exist_ok=True)
    # Remove only stale normalized files recorded in a prior generated manifest.
    # Unknown files in the output folder are left untouched.
    manifest_path = args.processed_dir / "dataset_manifest.csv"
    prior_generated: set[str] = set()
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as stream:
                prior_generated = {r.get("normalized_filename", "") for r in csv.DictReader(stream)}
        except (OSError, csv.Error):
            prior_generated = set()
    active_outputs = {f"{i['participant_id']}.csv" for i in results if i["eligible_for_training"]}
    for stale in prior_generated - active_outputs - {""}:
        candidate = eligible_dir / stale
        if candidate.is_file():
            candidate.unlink()

    for item in results:
        if item["eligible_for_training"]:
            _write_normalized(item, eligible_dir)
    # This is a script-owned scratch directory. Remove only its remaining files.
    for staged in staging_dir.glob("*.csv.tmp"):
        staged.unlink(missing_ok=True)
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    manifest_fields = [
        "participant_id", "source_filename", "normalized_filename", "delimiter_detected",
        "eeg_channel_count", "total_rows", "valid_eeg_rows", *MARKERS,
        "format_valid", "annotations_valid", "eligible_for_training", "exclusion_reason",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_row(item) for item in results)

    # Preserve raw annotation text for included and excluded files alike.
    annotation_path = args.processed_dir / "annotation_metadata.json"
    annotation_path.write_text(json.dumps({
        "markers": list(MARKERS),
        "files": [{
            "participant_id": item["participant_id"],
            "source_filename": item["source_filename"],
            "annotations": item["annotation_strings"],
        } for item in results],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = args.processed_dir / "preparation_report.md"
    report_path.write_text(render_report(results, delimiter_counts), encoding="utf-8")
    print(f"\nInspected {len(results)} files: {sum(i['eligible_for_training'] for i in results)} eligible, "
          f"{sum(not i['eligible_for_training'] for i in results)} excluded.")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(f"Annotations: {annotation_path}")
    print(f"Eligible EEG copies: {eligible_dir}")
    return 0 if len(results) == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
