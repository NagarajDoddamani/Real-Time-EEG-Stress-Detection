#!/usr/bin/env python3
"""Validate explicit condition marker locations without resolving duplicates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MARKERS = ("Baseline_start", "Baseline_end", "stress_start", "stress_end")
CONDITIONS = {
    "baseline": ("Baseline_start", "Baseline_end"),
    "stress": ("stress_start", "stress_end"),
}


def marker_occurrences(annotation_file: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Expand annotation metadata to one occurrence per marker and source row."""
    result = {marker: [] for marker in MARKERS}
    for annotation in annotation_file.get("annotations", []):
        for marker in annotation.get("markers", []):
            if marker not in result:
                continue
            for line in annotation.get("source_line_numbers", []):
                result[marker].append({
                    "source_line_number": int(line),
                    "timestamp": annotation.get("timestamp"),
                    "raw_text": annotation.get("raw_text", ""),
                    "field_name": annotation.get("field_name", ""),
                })
    for values in result.values():
        values.sort(key=lambda item: item["source_line_number"])
    return result


def validate_participant(annotation_file: dict[str, Any], total_rows: int) -> dict[str, Any]:
    """Validate marker completeness, uniqueness, sample bounds, order, and overlap.

    Source files have one header line. Thus physical line N maps to zero-based
    sample index N-2. This mapping is checked against normalized row counts.
    Intervals are half-open: [start_marker_sample, end_marker_sample).
    """
    occurrences = marker_occurrences(annotation_file)
    reasons: list[str] = []
    review: list[str] = []
    locations: dict[str, list[dict[str, Any]]] = {}
    sample_positions: dict[str, int] = {}
    for marker, values in occurrences.items():
        locations[marker] = values
        if not values:
            reasons.append(f"missing {marker}")
        elif len(values) > 1:
            lines = [v["source_line_number"] for v in values]
            reasons.append(f"duplicate {marker} at source lines {lines}; manual review required")
        else:
            line = values[0]["source_line_number"]
            index = line - 2
            if index < 0 or index >= total_rows:
                reasons.append(f"{marker} at source line {line} maps outside 0..{total_rows - 1}")
            else:
                sample_positions[marker] = index

    intervals: dict[str, dict[str, int]] = {}
    for condition, (start_marker, end_marker) in CONDITIONS.items():
        if start_marker in sample_positions and end_marker in sample_positions:
            start, end = sample_positions[start_marker], sample_positions[end_marker]
            if start >= end:
                reasons.append(f"{condition} start does not precede its end ({start} >= {end})")
            else:
                intervals[condition] = {"start_sample_index": start, "end_sample_index": end}

    if "baseline" in intervals and "stress" in intervals:
        baseline, stress = intervals["baseline"], intervals["stress"]
        if max(baseline["start_sample_index"], stress["start_sample_index"]) < min(
            baseline["end_sample_index"], stress["end_sample_index"]
        ):
            reasons.append("baseline and stress intervals overlap; manual review required")
        if stress["start_sample_index"] < baseline["start_sample_index"]:
            review.append("stress interval occurs before baseline; protocol order is not imposed")

    return {
        "participant_id": annotation_file.get("participant_id", ""),
        "source_filename": annotation_file.get("source_filename", ""),
        "total_rows": total_rows,
        "markers": locations,
        "intervals": intervals,
        "valid_for_windowing": not reasons,
        "exclusion_reasons": reasons,
        "manual_review_notes": review,
    }


def validate_dataset(metadata_path: Path, manifest_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {row["participant_id"]: row for row in csv.DictReader(stream)}
    results = []
    for item in metadata.get("files", []):
        participant = item.get("participant_id", "")
        row = manifest.get(participant)
        if row is None:
            results.append({
                "participant_id": participant,
                "source_filename": item.get("source_filename", ""),
                "total_rows": 0,
                "markers": marker_occurrences(item),
                "intervals": {},
                "valid_for_windowing": False,
                "exclusion_reasons": ["participant absent from dataset manifest"],
                "manual_review_notes": [],
            })
            continue
        validation = validate_participant(item, int(row.get("total_rows") or 0))
        if str(row.get("format_valid", "")).strip().casefold() != "true":
            validation["exclusion_reasons"].append("recording format is invalid according to dataset manifest")
            validation["valid_for_windowing"] = False
        if str(row.get("eligible_for_training", "")).strip().casefold() != "true":
            validation["exclusion_reasons"].append("recording is not eligible in dataset manifest")
            validation["valid_for_windowing"] = False
        results.append(validation)
    return {"markers": list(MARKERS), "participants": results}


def render_report(report: dict[str, Any]) -> str:
    participants = report["participants"]
    valid = [p for p in participants if p["valid_for_windowing"]]
    lines = [
        "# Condition marker validation report", "",
        f"- Participants inspected: **{len(participants)}**",
        f"- Marker-valid for windowing: **{len(valid)}**",
        f"- Blocked for missing, duplicate, invalid-order, out-of-bounds, or overlapping markers: **{len(participants)-len(valid)}**",
        "- Marker row mapping: source physical line minus 2 gives zero-based sample index (one header line).",
        "- Intervals use the half-open convention [start marker sample, end marker sample).",
        "- Annotation timestamps are preserved verbatim; they are not used to infer sampling frequency or duration.", "",
        "| Participant | Baseline interval (samples) | Stress interval (samples) | Status | Details |",
        "|---|---:|---:|---|---|",
    ]
    for p in participants:
        b = p["intervals"].get("baseline", {})
        s = p["intervals"].get("stress", {})
        btxt = f"[{b['start_sample_index']}, {b['end_sample_index']})" if b else "unresolved"
        stxt = f"[{s['start_sample_index']}, {s['end_sample_index']})" if s else "unresolved"
        details = p["exclusion_reasons"] + p["manual_review_notes"]
        lines.append(f"| {p['participant_id']} | {btxt} | {stxt} | {'valid' if p['valid_for_windowing'] else 'blocked'} | {'; '.join(details) or '—'} |")
    lines += ["", "## Marker evidence", ""]
    for p in participants:
        lines += [f"### {p['participant_id']}", ""]
        for marker in MARKERS:
            hits = p["markers"].get(marker, [])
            evidence = "; ".join(
                f"line {hit['source_line_number']} / sample {hit['source_line_number']-2}; timestamp={hit['timestamp'] or 'absent'}; text={hit['raw_text']!r}"
                for hit in hits
            ) or "missing"
            lines.append(f"- {marker}: {evidence}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/annotation_metadata.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    report = validate_dataset(args.metadata, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "marker_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "marker_validation_report.md").write_text(render_report(report), encoding="utf-8")
    for item in report["participants"]:
        print(f"{item['participant_id']}: {'VALID' if item['valid_for_windowing'] else 'BLOCKED'}" +
              (f" — {'; '.join(item['exclusion_reasons'])}" if item["exclusion_reasons"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
