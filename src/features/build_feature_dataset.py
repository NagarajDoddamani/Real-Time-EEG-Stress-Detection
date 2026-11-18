#!/usr/bin/env python3
"""Build participant-labelled training features from validated EEG windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE = ROOT / "src" / "data"
FEATURE_SOURCE = ROOT / "src" / "features"
for source in (str(DATA_SOURCE), str(FEATURE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

from create_windows import window_indices
from window_features import extract_window_features, feature_names

WINDOW_FIELDS = [
    "participant_id", "condition", "label", "source_filename", "normalized_filename",
    "start_sample_index", "end_sample_index_exclusive",
    "condition_start_marker_timestamp", "condition_end_marker_timestamp",
]
FEATURE_METADATA_FIELDS = {
    "participant_id", "condition", "label", "source_filename",
    "start_sample_index", "end_sample_index_exclusive",
}
CONDITIONS = (("baseline", "Baseline_start", "Baseline_end"), ("stress", "stress_start", "stress_end"))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_signal(path: Path, channels: list[str]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if header != channels:
        raise ValueError(f"{path.name}: normalized header/order does not match configured 19 EEG channels")
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}: could not parse normalized samples: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != len(channels) or not len(values):
        raise ValueError(f"{path.name}: expected non-empty samples x {len(channels)} channel matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: contains NaN or infinite values; refusing silent row removal")
    return values


def build_dataset(config: dict[str, Any], marker_report: dict[str, Any],
                  manifest: dict[str, dict[str, str]], eligible_dir: Path
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fs = float(config["sampling_frequency_hz"])
    duration = float(config["window_duration_seconds"])
    overlap = float(config["window_overlap_fraction"])
    channels = config["eeg_channels"]
    bands = config["frequency_bands_hz"]
    label_map = config["label_map"]
    n_samples = int(round(fs * duration))
    if n_samples != 2000 or int(round(n_samples * (1.0 - overlap))) != 1000:
        raise ValueError("configured settings must yield 2000-sample windows and a 1000-sample step")
    for band, bounds in bands.items():
        if len(bounds) != 2 or not 0 <= float(bounds[0]) < float(bounds[1]) < fs / 2:
            raise ValueError(f"{band} band must be below Nyquist ({fs / 2:g} Hz)")

    feature_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for participant in marker_report.get("participants", []):
        participant_id = participant["participant_id"]
        source_filename = participant.get("source_filename", "")
        if not participant.get("valid_for_windowing"):
            reason = "; ".join(participant.get("exclusion_reasons", [])) or "marker validation failed"
            for condition, _, _ in CONDITIONS:
                coverage_rows.append({"participant_id": participant_id, "condition": condition,
                                      "window_count": 0, "interval_samples": 0, "status": "skipped", "reason": reason})
            continue

        source_row = manifest.get(participant_id)
        if not source_row or str(source_row.get("eligible_for_training", "")).casefold() != "true":
            raise ValueError(f"{participant_id}: marker report says valid but preparation manifest does not mark it eligible")
        normalized_name = source_row.get("normalized_filename", "")
        if not normalized_name:
            raise ValueError(f"{participant_id}: normalized filename missing from preparation manifest")
        signal = load_signal(eligible_dir / normalized_name, channels)
        if len(signal) != int(source_row["total_rows"]):
            raise ValueError(f"{participant_id}: normalized sample count disagrees with manifest")

        for condition, start_marker, end_marker in CONDITIONS:
            interval = participant["intervals"][condition]
            start_i, end_i = int(interval["start_sample_index"]), int(interval["end_sample_index"])
            ranges = window_indices(start_i, end_i, fs, duration, overlap)
            reason = "" if ranges else f"condition interval has {end_i - start_i} samples; needs at least {n_samples}"
            coverage_rows.append({"participant_id": participant_id, "condition": condition,
                                  "window_count": len(ranges), "interval_samples": end_i - start_i,
                                  "status": "included" if ranges else "skipped", "reason": reason})
            start_stamp = participant["markers"][start_marker][0].get("timestamp") or ""
            end_stamp = participant["markers"][end_marker][0].get("timestamp") or ""
            for start, end in ranges:
                if end - start != n_samples:
                    raise RuntimeError("window generator returned a non-configured window size")
                sample_window = signal[start:end]
                features = extract_window_features(sample_window, channels, config)
                window_rows.append({
                    "participant_id": participant_id, "condition": condition,
                    "label": int(label_map[condition]), "source_filename": source_filename,
                    "normalized_filename": normalized_name,
                    "start_sample_index": start, "end_sample_index_exclusive": end,
                    "condition_start_marker_timestamp": start_stamp,
                    "condition_end_marker_timestamp": end_stamp,
                })
                feature_rows.append({
                    "participant_id": participant_id, "condition": condition,
                    "label": int(label_map[condition]), "source_filename": source_filename,
                    "start_sample_index": start, "end_sample_index_exclusive": end,
                    **features,
                })

    if not feature_rows:
        raise ValueError("no feature windows generated")
    return feature_rows, window_rows, coverage_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def convert(config_path: Path, marker_report_path: Path, manifest_path: Path,
            eligible_dir: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fs = config.get("sampling_frequency_hz")
    if not config.get("sampling_frequency_verified") or fs is None:
        raise ValueError("training features require the verified sampling frequency in eeg_config.json")
    fs = float(fs)
    if not math.isfinite(fs) or fs != 500.0:
        raise ValueError("this dataset task is configured for the user-confirmed 500 Hz recordings")
    config["sampling_frequency_hz"] = fs
    marker_report = json.loads(marker_report_path.read_text(encoding="utf-8"))
    manifest = {row["participant_id"]: row for row in read_csv_dicts(manifest_path)}
    features, windows, coverage = build_dataset(config, marker_report, manifest, eligible_dir)

    expected_feature_names = feature_names(config["eeg_channels"], config["frequency_bands_hz"])
    if list(features[0])[-len(expected_feature_names):] != expected_feature_names:
        raise RuntimeError("generated feature order differs from the configured schema")
    files = [output_dir / name for name in (
        "training_features.csv", "window_index.csv", "window_coverage.csv", "feature_manifest.json")]
    existing = [str(path) for path in files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("output exists; pass --overwrite to replace: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_features.csv", features,
              ["participant_id", "condition", "label", "source_filename",
               "start_sample_index", "end_sample_index_exclusive", *expected_feature_names])
    from create_windows import WINDOW_FIELDS
    write_csv(output_dir / "window_index.csv", windows, WINDOW_FIELDS)
    write_csv(output_dir / "window_coverage.csv", coverage,
              ["participant_id", "condition", "window_count", "interval_samples", "status", "reason"])
    metadata = {
        "format": "verified_sampling_rate_training_features",
        "sampling_frequency_hz": fs,
        "sampling_frequency_status": "verified_by_user_for_existing_recordings",
        "signal_units": config.get("signal_units"),
        "signal_units_status": "unverified",
        "channel_order": config["eeg_channels"],
        "feature_names": expected_feature_names,
        "feature_count": len(expected_feature_names),
        "feature_definitions": {
            "time_domain": ["mean", "sample standard deviation", "sample variance", "RMS", "peak-to-peak"],
            "frequency_domain": "Welch band power and relative power for every configured band",
            "spectral_entropy": "Normalized Shannon entropy of Welch PSD bins from 0.5 Hz through the highest configured band edge",
        },
        "window_duration_seconds": config["window_duration_seconds"],
        "window_samples": 2000,
        "window_overlap_fraction": config["window_overlap_fraction"],
        "window_step_samples": 1000,
        "frequency_bands_hz": config["frequency_bands_hz"],
        "psd_configuration": config["psd"],
        "label_map": config["label_map"],
        "participant_ids": sorted({row["participant_id"] for row in features}, key=str.casefold),
        "window_count": len(features),
        "class_window_counts": {str(label): sum(int(row["label"]) == label for row in features)
                                for label in sorted(set(int(v) for v in config["label_map"].values()))},
        "deployment_ready": False,
        "limitations": ["labels are experimental conditions, not clinical stress diagnoses",
                        "physical EEG units and export scaling are unverified",
                        "no filtering was applied; exported raw-data filter behavior is unverified"],
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eeg_config.json")
    parser.add_argument("--marker-report", type=Path, default=ROOT / "data" / "processed" / "marker_validation.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "dataset_manifest.csv")
    parser.add_argument("--eligible-dir", type=Path, default=ROOT / "data" / "processed" / "eligible")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "features" / "training_500Hz_19ch")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = convert(args.config, args.marker_report, args.manifest, args.eligible_dir,
                         args.output_dir, args.overwrite)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(2, f"Feature preparation failed: {exc}\n")
    print(f"Prepared {result['window_count']} windows with {result['feature_count']} features each "
          f"from {len(result['participant_ids'])} participants.")
    print(f"Class window counts: {result['class_window_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
