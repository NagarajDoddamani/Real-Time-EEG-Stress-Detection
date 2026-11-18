from __future__ import annotations

import sys
import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "training"))
sys.path.insert(0, str(ROOT / "src" / "realtime"))

from validate_markers import validate_participant
from create_windows import config_sampling_rate, window_indices
from frequency_features import require_verified_sampling_frequency
from statistical_features import extract_statistical_features
from participant_split import split_participants
from train_models import evaluate_loso, train, validate_training_inputs
from summarize_predictions import summarize_rows
from predict_all_recordings import load_raw_recording
from window_features import extract_window_features, feature_names
from build_feature_dataset import build_dataset, load_signal
from recorded_replay import replay


def participant(positions: dict[str, list[int]], total: int = 100) -> dict:
    keys = ("Baseline_start", "Baseline_end", "stress_start", "stress_end")
    annotations = []
    for key in keys:
        for line in positions.get(key, []):
            annotations.append({"markers": [key], "source_line_numbers": [line],
                                "timestamp": None, "raw_text": key, "field_name": "event"})
    return {"participant_id": "p1", "source_filename": "p1.csv", "annotations": annotations}


class MarkerValidationTests(unittest.TestCase):
    def test_valid_markers_map_to_half_open_intervals(self):
        item = participant({"Baseline_start": [2], "Baseline_end": [32], "stress_start": [42], "stress_end": [72]})
        result = validate_participant(item, 100)
        self.assertTrue(result["valid_for_windowing"])
        self.assertEqual(result["intervals"]["baseline"], {"start_sample_index": 0, "end_sample_index": 30})

    def test_duplicate_marker_blocks_without_selecting_one(self):
        item = participant({"Baseline_start": [2, 3], "Baseline_end": [32], "stress_start": [42], "stress_end": [72]})
        result = validate_participant(item, 100)
        self.assertFalse(result["valid_for_windowing"])
        self.assertNotIn("baseline", result["intervals"])
        self.assertTrue(any("duplicate Baseline_start" in reason for reason in result["exclusion_reasons"]))

    def test_bad_chronology_and_out_of_bounds_are_rejected(self):
        item = participant({"Baseline_start": [40], "Baseline_end": [20], "stress_start": [102], "stress_end": [110]})
        result = validate_participant(item, 100)
        self.assertFalse(result["valid_for_windowing"])
        self.assertTrue(any("does not precede" in reason for reason in result["exclusion_reasons"]))
        self.assertTrue(any("outside" in reason for reason in result["exclusion_reasons"]))

    def test_overlapping_conditions_are_rejected(self):
        item = participant({"Baseline_start": [2], "Baseline_end": [52], "stress_start": [42], "stress_end": [72]})
        result = validate_participant(item, 100)
        self.assertFalse(result["valid_for_windowing"])
        self.assertTrue(any("overlap" in reason for reason in result["exclusion_reasons"]))


class ProcessingTests(unittest.TestCase):
    def test_unverified_sampling_rate_blocks_window_and_frequency_work(self):
        config = {"sampling_frequency_hz": None, "sampling_frequency_verified": False,
                  "frequency_dependent_processing_permitted": False}
        with self.assertRaisesRegex(ValueError, "unverified"):
            config_sampling_rate(config)
        with self.assertRaisesRegex(ValueError, "verified"):
            require_verified_sampling_frequency(config)

    def test_windows_stay_inside_interval(self):
        windows = window_indices(10, 35, 10, 1.0, 0.5)
        self.assertEqual(windows, [(10, 20), (15, 25), (20, 30), (25, 35)])
        self.assertTrue(all(start >= 10 and end <= 35 for start, end in windows))

    def test_statistics_and_channel_order_are_explicit(self):
        result = extract_statistical_features([[1, 4], [3, 8]], ["Fp1", "Fp2"])
        self.assertEqual(result["Fp1_mean"], 2)
        self.assertEqual(result["Fp2_mean"], 6)
        with self.assertRaisesRegex(ValueError, "channel count"):
            extract_statistical_features([[1]], ["Fp1", "Fp2"])

    def test_participant_splits_are_disjoint_and_deterministic(self):
        ids = [f"p{i}" for i in range(10)]
        first = split_participants(ids, seed=7)
        self.assertEqual(first, split_participants(ids, seed=7))
        sets = [set(first[key]) for key in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(ids))

    def test_split_rejects_insufficient_participants(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            split_participants(["p1", "p2"])

    def test_training_is_blocked_until_metadata_windows_and_features_are_ready(self):
        config = {"sampling_frequency_hz": None, "sampling_frequency_verified": False,
                  "signal_units": None, "label_map": {"baseline": 0, "stress": 1}}
        markers = {"participants": [{"participant_id": "p1", "valid_for_windowing": True}]}
        self.assertIn("sampling frequency is not verified",
                      validate_training_inputs(config, markers, [], [])[0])
        self.assertTrue(any("window index is empty" in item for item in
                            validate_training_inputs(config, markers, [], [])))
        self.assertTrue(any("feature table is empty" in item for item in
                            validate_training_inputs(config, markers, [], [])))

    def test_feature_rows_must_match_windows_and_configured_labels(self):
        config = {"sampling_frequency_hz": 500, "sampling_frequency_verified": True,
                  "signal_units": None, "label_map": {"baseline": 0, "stress": 1}}
        markers = {"participants": [{"participant_id": "p1", "valid_for_windowing": True}]}
        window = {"participant_id": "p1", "condition": "baseline", "label": "0",
                  "source_filename": "p1.csv",
                  "start_sample_index": "0", "end_sample_index_exclusive": "2000"}
        feature = dict(window, Fp1_alpha_power="2.0")
        self.assertEqual(validate_training_inputs(config, markers, [window], [feature]), [])
        bad = dict(feature, label="1")
        errors = validate_training_inputs(config, markers, [window], [bad])
        self.assertTrue(any("correspond exactly" in item for item in errors))
        self.assertTrue(any("condition/label" in item for item in errors))


class FeatureAndLossoTests(unittest.TestCase):
    def setUp(self):
        self.channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                         "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
        self.config = {
            "eeg_channels": self.channels,
            "sampling_frequency_hz": 500,
            "sampling_frequency_verified": True,
            "frequency_dependent_processing_permitted": True,
            "signal_units": None,
            "window_duration_seconds": 4.0,
            "window_overlap_fraction": 0.5,
            "frequency_bands_hz": {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13],
                                    "beta": [13, 30], "gamma": [30, 80]},
            "psd": {"segment_duration_seconds": 2.0, "overlap_fraction": 0.5},
            "label_map": {"baseline": 0, "stress": 1},
        }

    def test_19_channel_feature_schema_and_finite_values(self):
        t = np.arange(2000) / 500
        signal = np.column_stack([np.sin(2 * np.pi * (8 + i % 5) * t) for i in range(19)])
        result = extract_window_features(signal, self.channels, self.config)
        schema = feature_names(self.channels, self.config["frequency_bands_hz"])
        self.assertEqual(len(schema), 304)
        self.assertEqual(list(result), schema)
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertIn("Fp1_variance", result)
        self.assertIn("O2_gamma_relative_power", result)
        self.assertIn("Cz_spectral_entropy", result)

    def test_invalid_signal_values_are_rejected(self):
        values = np.ones((2000, 19))
        values[17, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            extract_window_features(values, self.channels, self.config)

    def test_feature_dataset_assigns_event_labels_without_boundary_crossing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p1.csv"
            samples = np.random.default_rng(3).normal(size=(8000, 19))
            np.savetxt(path, samples, delimiter=",", header=",".join(self.channels), comments="")
            marker_report = {"participants": [{
                "participant_id": "p1", "source_filename": "p1.csv", "valid_for_windowing": True,
                "intervals": {"baseline": {"start_sample_index": 0, "end_sample_index": 4000},
                              "stress": {"start_sample_index": 4000, "end_sample_index": 8000}},
                "markers": {"Baseline_start": [{"timestamp": "t0"}],
                            "Baseline_end": [{"timestamp": "t1"}],
                            "stress_start": [{"timestamp": "t2"}],
                            "stress_end": [{"timestamp": "t3"}]},
            }]}
            manifest = {"p1": {"eligible_for_training": "True", "normalized_filename": "p1.csv",
                               "total_rows": "8000"}}
            features, windows, coverage = build_dataset(self.config, marker_report, manifest, Path(tmp))
            self.assertEqual(len(features), 6)
            self.assertEqual([int(row["label"]) for row in features], [0, 0, 0, 1, 1, 1])
            self.assertTrue(all(int(w["end_sample_index_exclusive"]) - int(w["start_sample_index"]) == 2000
                                for w in windows))
            self.assertTrue(all((int(w["start_sample_index"]) < 4000) == (w["condition"] == "baseline")
                                for w in windows))
            self.assertEqual([row["window_count"] for row in coverage], [3, 3])

    def test_loader_rejects_bad_channel_header_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text(",".join(self.channels[:-1]) + "\n" + ",".join(["1"] * 18) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header/order"):
                load_signal(path, self.channels)
            path.write_text(",".join(self.channels) + "\n" + ",".join(["nan"] + ["1"] * 18) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                load_signal(path, self.channels)

    def test_4_second_windows_are_2000_samples_with_1000_step(self):
        windows = window_indices(0, 4000, 500, 4.0, 0.5)
        self.assertEqual(windows, [(0, 2000), (1000, 3000), (2000, 4000)])
        self.assertTrue(all(end - start == 2000 for start, end in windows))

    def test_recorded_replay_emits_the_same_19_channel_schema(self):
        t = np.arange(2000) / 500
        samples = np.column_stack([np.sin(2 * np.pi * (8 + i % 5) * t) for i in range(19)])
        rows = replay(samples, self.channels, self.config, 500, 64, 1.0, True,
                      start_sample_index=345)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["window_start_sample"], 345)
        self.assertEqual(rows[0]["window_start_seconds_at_replay_rate"], 0.69)
        self.assertEqual([name for name in feature_names(self.channels, self.config["frequency_bands_hz"])
                          if name in rows[0]], feature_names(self.channels, self.config["frequency_bands_hz"]))

    def test_loso_scaler_is_fitted_only_on_training_participants(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        groups = np.repeat(["p1", "p2", "p3"], 2)
        X = np.asarray([[1, 0], [1, 1], [10, 0], [10, 1], [100, 0], [100, 1]], dtype=float)
        y = np.tile([0, 1], 3)
        rows = [{"participant_id": p, "condition": c, "start_sample_index": "0",
                 "end_sample_index_exclusive": "2000"}
                for p, c in zip(groups, ["baseline", "stress"] * 3)]
        fitted = []

        def factory():
            model = Pipeline([("scaler", StandardScaler()),
                              ("model", LogisticRegression(max_iter=500, random_state=1))])
            fitted.append(model)
            return model

        evaluate_loso(X, y, groups, rows, {"test_pipeline": factory})
        for model, held_out in zip(fitted, ["p1", "p2", "p3"]):
            expected = X[groups != held_out].mean(axis=0)
            np.testing.assert_allclose(model.named_steps["scaler"].mean_, expected)

    def test_final_pipeline_saves_and_reloads_with_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            marker_path = root / "markers.json"
            feature_path = root / "features.csv"
            window_path = root / "windows.csv"
            manifest_path = root / "feature_manifest.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            participants = ["p1", "p2", "p3"]
            marker_path.write_text(json.dumps({"participants": [
                {"participant_id": p, "valid_for_windowing": True} for p in participants
            ]}), encoding="utf-8")
            names = ["feat_a", "feat_b"]
            manifest_path.write_text(json.dumps({
                "sampling_frequency_status": "verified_by_user_for_existing_recordings",
                "sampling_frequency_hz": 500, "channel_order": self.channels,
                "feature_names": names,
                "feature_definitions": {}, "frequency_bands_hz": self.config["frequency_bands_hz"],
                "psd_configuration": self.config["psd"],
            }), encoding="utf-8")
            windows, features = [], []
            for participant_index, participant_id in enumerate(participants):
                for condition, label in (("baseline", 0), ("stress", 1)):
                    start = label * 2000
                    row = {"participant_id": participant_id, "condition": condition, "label": label,
                           "source_filename": participant_id + ".csv", "start_sample_index": start,
                           "end_sample_index_exclusive": start + 2000}
                    windows.append(row)
                    features.append({**row, "feat_a": label + participant_index * 0.1,
                                     "feat_b": label * 2 + participant_index * 0.05})
            for path, rows in ((window_path, windows), (feature_path, features)):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
            output_dir = root / "models"
            result = train(feature_path, config_path, marker_path, window_path,
                           manifest_path, output_dir)
            import joblib
            loaded = joblib.load(output_dir / "stress_model.pkl")
            X = np.asarray([[float(row[name]) for name in names] for row in features])
            self.assertEqual(result["participant_count"], 3)
            np.testing.assert_array_equal(loaded.predict(X), joblib.load(output_dir / "stress_model.pkl").predict(X))
            self.assertTrue((output_dir / "evaluation" / "out_of_fold_predictions.csv").exists())

    def test_oof_prediction_summary_is_condition_and_participant_grouped(self):
        rows = [
            {"model": "svm_rbf", "participant_id": "p1", "condition": "baseline", "true_label": "0",
             "predicted_label": "0", "probability_stress": "0.2"},
            {"model": "svm_rbf", "participant_id": "p1", "condition": "baseline", "true_label": "0",
             "predicted_label": "1", "probability_stress": "0.7"},
            {"model": "svm_rbf", "participant_id": "p1", "condition": "stress", "true_label": "1",
             "predicted_label": "1", "probability_stress": "0.8"},
        ]
        summary = summarize_rows(rows, "svm_rbf")
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["actual_condition"], "baseline")
        self.assertEqual(summary[0]["window_accuracy"], 0.5)
        self.assertEqual(summary[1]["majority_predicted_condition"], "stress")
        self.assertAlmostEqual(summary[1]["mean_probability_stress"], 0.8)

    def test_raw_inference_loader_maps_headers_and_recovers_row_width(self):
        channels = ["Fp1", "Fp2", "F7"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text("Fp1;Fp2;F7;REF;event\n1;2;3;4;Baseline_start\n5;6;7;8\n", encoding="utf-8")
            samples, delimiter, width_issues = load_raw_recording(path, channels)
        self.assertEqual(delimiter, "semicolon")
        self.assertEqual(width_issues, 1)
        np.testing.assert_array_equal(samples, [[1, 2, 3], [5, 6, 7]])


if __name__ == "__main__":
    unittest.main()
