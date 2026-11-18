#!/usr/bin/env python3
"""Run the saved feature-compatible model over every numeric raw recording.

This is batch inference, not an evaluation. Rows from training participants
are in-sample. Marker-invalid recordings have no condition ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[2]
from src.features.window_features import extract_window_features, feature_names as canonical_feature_names


def load_raw_recording(path: Path, channels: list[str]) -> tuple[np.ndarray, str, int]:
    """Read channel columns by header name; auxiliary columns are ignored."""
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        header_line = stream.readline()
        if not header_line:
            raise ValueError(f"{path.name}: empty recording")
        if header_line.count(";") == header_line.count(","):
            raise ValueError(f"{path.name}: cannot uniquely determine comma/semicolon delimiter")
        delimiter = ";" if header_line.count(";") > header_line.count(",") else ","
        stream.seek(0)
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        headers = [item.strip().lstrip("\ufeff") for item in next(reader)]
        lowered = [item.casefold() for item in headers]
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"{path.name}: duplicate CSV headers")
        missing = [channel for channel in channels if channel.casefold() not in lowered]
        if missing:
            raise ValueError(f"{path.name}: missing required EEG channels: {missing}")
        positions = [lowered.index(channel.casefold()) for channel in channels]
        max_position = max(positions)
        sample_rows: list[list[float]] = []
        recoverable_width_rows = 0
        for physical_line, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                recoverable_width_rows += 1
            if len(row) <= max_position:
                raise ValueError(f"{path.name}: line {physical_line} lacks one or more EEG columns")
            try:
                values = [float(row[position].strip()) for position in positions]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path.name}: invalid EEG value at line {physical_line}: {exc}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path.name}: non-finite EEG value at line {physical_line}")
            sample_rows.append(values)
    if not sample_rows:
        raise ValueError(f"{path.name}: no EEG samples found")
    return np.asarray(sample_rows, dtype=np.float64), "semicolon" if delimiter == ";" else "comma", recoverable_width_rows


def annotate_window(start: int, end: int, participant: dict[str, Any]) -> str:
    if not participant.get("valid_for_windowing"):
        return "unlabelled_marker_invalid"
    intervals = participant.get("intervals", {})
    for condition in ("baseline", "stress"):
        interval = intervals.get(condition, {})
        if start >= int(interval.get("start_sample_index", -1)) and end <= int(interval.get("end_sample_index", -1)):
            return condition
    return "transition_or_outside_labelled_interval"


def predict_dataset(raw_dir: Path, manifest_path: Path, marker_path: Path,
                    config_path: Path, model_path: Path, model_metadata_path: Path,
                    output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    channels = config["eeg_channels"]
    bands = config["frequency_bands_hz"]
    fs = float(config["sampling_frequency_hz"])
    if not config.get("sampling_frequency_verified") or fs != 500:
        raise ValueError("batch model prediction requires the configured verified 500 Hz rate")
    expected_names = canonical_feature_names(channels, bands)
    if model_metadata.get("channel_names_and_order") != channels:
        raise ValueError("model channel order differs from the current EEG configuration")
    if model_metadata.get("feature_names_and_order") != expected_names:
        raise ValueError("model feature schema differs from the current feature extractor")
    model = joblib.load(model_path)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {row["participant_id"]: row for row in csv.DictReader(stream)}
    marker_report = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_by_id = {item["participant_id"]: item for item in marker_report.get("participants", [])}
    participants_trained = set(model_metadata.get("training_participant_ids", []))
    source_files = sorted(raw_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if not source_files:
        raise ValueError(f"no CSV recordings found in {raw_dir}")

    predictions_path = output_dir / "all_recordings_window_predictions.csv"
    summary_path = output_dir / "all_recordings_participant_summary.csv"
    metadata_path = output_dir / "prediction_run_metadata.json"
    paths = [predictions_path, summary_path, metadata_path]
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("prediction outputs exist; pass --overwrite to replace: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    window_fields = [
        "participant_id", "source_filename", "participant_fit_status", "marker_status",
        "annotation_condition", "start_sample_index", "end_sample_index_exclusive",
        "predicted_label", "predicted_condition", "probability_baseline", "probability_stress",
    ]
    summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    participant_metadata: list[dict[str, Any]] = []
    total_width_issues = 0
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=window_fields, lineterminator="\n")
        writer.writeheader()
        for source_path in source_files:
            participant_id = source_path.stem
            print(f"Predicting {participant_id}...", flush=True)
            manifest_row = manifest.get(participant_id)
            if manifest_row is None or str(manifest_row.get("format_valid", "")).casefold() != "true":
                raise ValueError(f"{participant_id}: no format-valid dataset manifest row")
            data, delimiter, width_issues = load_raw_recording(source_path, channels)
            if len(data) != int(manifest_row["total_rows"]) or len(data) != int(manifest_row["valid_eeg_rows"]):
                raise ValueError(f"{participant_id}: raw row count differs from the audited manifest")
            total_width_issues += width_issues
            marker = marker_by_id.get(participant_id, {"valid_for_windowing": False})
            marker_status = "valid_unique_intervals" if marker.get("valid_for_windowing") else (
                "ambiguous_duplicate_markers" if any("duplicate" in reason for reason in marker.get("exclusion_reasons", []))
                else "incomplete_or_invalid_markers"
            )
            fit_status = "training_participant_in_sample" if participant_id in participants_trained else "not_in_model_training"
            window_size = int(round(fs * float(config["window_duration_seconds"])))
            step = int(round(window_size * (1.0 - float(config["window_overlap_fraction"]))))
            participant_rows: list[dict[str, Any]] = []
            feature_matrix: list[list[float]] = []
            for start in range(0, len(data) - window_size + 1, step):
                end = start + window_size
                features = extract_window_features(data[start:end], channels, config)
                feature_matrix.append([features[name] for name in expected_names])
                participant_rows.append({
                    "participant_id": participant_id,
                    "source_filename": source_path.name,
                    "participant_fit_status": fit_status,
                    "marker_status": marker_status,
                    "annotation_condition": annotate_window(start, end, marker),
                    "start_sample_index": start,
                    "end_sample_index_exclusive": end,
                })
            if not feature_matrix:
                raise ValueError(f"{participant_id}: recording is shorter than one {window_size}-sample window")
            X = np.asarray(feature_matrix, dtype=np.float64)
            if not np.isfinite(X).all():
                raise ValueError(f"{participant_id}: extracted feature matrix contains non-finite values")
            predicted = np.asarray(model.predict(X), dtype=int)
            probabilities = np.asarray(model.predict_proba(X), dtype=float)
            classes = [int(value) for value in model.classes_]
            for index, record in enumerate(participant_rows):
                record["predicted_label"] = int(predicted[index])
                record["predicted_condition"] = {int(v): k for k, v in config["label_map"].items()}.get(int(predicted[index]), "unknown")
                record["probability_baseline"] = float(probabilities[index, classes.index(0)]) if 0 in classes else 0.0
                record["probability_stress"] = float(probabilities[index, classes.index(1)]) if 1 in classes else 0.0
                writer.writerow(record)
                summary_groups[participant_id].append(record)
            participant_metadata.append({
                "participant_id": participant_id, "source_filename": source_path.name,
                "rows": len(data), "windows": len(participant_rows), "delimiter": delimiter,
                "recoverable_row_width_mismatches": width_issues,
                "participant_fit_status": fit_status, "marker_status": marker_status,
            })
            stream.flush()
            print(f"  {len(participant_rows)} windows ({fit_status}; {marker_status})", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for participant_id, rows in sorted(summary_groups.items(), key=lambda item: item[0].casefold()):
        baseline_count = sum(row["predicted_label"] == 0 for row in rows)
        stress_count = sum(row["predicted_label"] == 1 for row in rows)
        summary_rows.append({
            "participant_id": participant_id,
            "source_filename": rows[0]["source_filename"],
            "participant_fit_status": rows[0]["participant_fit_status"],
            "marker_status": rows[0]["marker_status"],
            "windows_predicted": len(rows),
            "predicted_baseline_windows": baseline_count,
            "predicted_stress_windows": stress_count,
            "majority_predicted_condition": "baseline" if baseline_count > stress_count else "stress" if stress_count > baseline_count else "tie",
            "mean_probability_stress": statistics.fmean(row["probability_stress"] for row in rows),
        })
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "model_algorithm": model_metadata["model_algorithm"],
        "model_path": str(model_path),
        "sampling_frequency_hz": fs,
        "sampling_rate_status": model_metadata.get("sampling_frequency_status"),
        "window_samples": int(round(fs * float(config["window_duration_seconds"]))),
        "window_step_samples": int(round(fs * float(config["window_duration_seconds"]) * (1.0 - float(config["window_overlap_fraction"])))),
        "channel_order": channels,
        "feature_count": len(expected_names),
        "recordings_processed": len(participant_metadata),
        "windows_predicted": sum(item["windows"] for item in participant_metadata),
        "recoverable_row_width_mismatches": total_width_issues,
        "participant_training_membership": sorted(participants_trained, key=str.casefold),
        "recording_details": participant_metadata,
        "outputs": {"window_predictions": str(predictions_path), "participant_summary": str(summary_path)},
        "interpretation": "Model outputs are experimental-condition class predictions, not clinical stress diagnoses.",
        "evaluation_warning": "This full-fit inference output is not an accuracy evaluation. Training participants are in-sample; marker-invalid participants have no condition ground truth. Use models/evaluation/out_of_fold_predictions.csv for held-out evaluation of the 13 valid participants.",
        "units_status": "unverified",
        "deployment_ready": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "dataset_manifest.csv")
    parser.add_argument("--marker-report", type=Path, default=ROOT / "data" / "processed" / "marker_validation.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eeg_config.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "stress_model.pkl")
    parser.add_argument("--model-metadata", type=Path, default=ROOT / "models" / "model_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "prediction_runs" / "all_recordings_500Hz")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = predict_dataset(args.raw_dir, args.manifest, args.marker_report, args.config,
                                 args.model, args.model_metadata, args.output_dir, args.overwrite)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Dataset prediction failed: {exc}\n")
    print(f"Predicted {result['windows_predicted']} windows across {result['recordings_processed']} recordings.")
    print(f"Window output: {result['outputs']['window_predictions']}")
    print(f"Participant summary: {result['outputs']['participant_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
