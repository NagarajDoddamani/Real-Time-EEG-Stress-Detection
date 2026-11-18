from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.acquisition.pipeline import (RollingWindowBuffer, TrainedModel, estimate_rate,
                                      extract_stream_metadata, map_eeg_channels)
from src.acquisition.recording import LocalRecording
from src.acquisition.cli import choose_stream, connect
from src.features.window_features import extract_window_features


class _Node:
    def __init__(self, values: list[str], index: int = 0):
        self.values, self.index = values, index
    def child(self, name: str):
        return self if name in {"channels", "channel"} else self
    def child_value(self, name: str):
        return self.values[self.index] if name == "label" and self.index < len(self.values) else ""
    def next_sibling(self):
        return _Node(self.values, self.index + 1)


class _Info:
    def __init__(self, labels: list[str]): self.labels = labels
    def name(self): return "simulated EEG"
    def type(self): return "EEG"
    def source_id(self): return "simulator-01"
    def channel_count(self): return len(self.labels)
    def nominal_srate(self): return 500.0
    def channel_format(self): return "cf_float32"
    def uid(self): return "uid-1"
    def desc(self): return _Node(self.labels)


class AcquisitionPipelineTests(unittest.TestCase):
    def test_env_loader_reads_quotes_and_preserves_existing_shell_values(self):
        from src.storage.neo4j_store import load_project_env
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text('EEG_TEST_ENV="from-file"\nEEG_EXISTING_ENV=ignored\n', encoding="utf-8")
            with patch.dict("os.environ", {"EEG_EXISTING_ENV": "from-shell"}, clear=False):
                import os
                os.environ.pop("EEG_TEST_ENV", None)
                load_project_env(env_path)
                self.assertEqual(os.environ["EEG_TEST_ENV"], "from-file")
                self.assertEqual(os.environ["EEG_EXISTING_ENV"], "from-shell")
                os.environ.pop("EEG_TEST_ENV", None)

    def test_metadata_extraction_and_label_mapping_include_extras(self):
        names = ["Fp1", "Fp2", "REF"]
        meta = extract_stream_metadata(_Info(names))
        self.assertEqual(meta.channel_count, 3)
        self.assertEqual(meta.channel_labels, names)
        self.assertEqual(map_eeg_channels(names, ["Fp2", "Fp1"]), {"Fp2": 1, "Fp1": 0})

    def test_missing_and_duplicate_labels_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            map_eeg_channels(["Fp1", "REF"], ["Fp1", "Fp2"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            map_eeg_channels(["Fp1", "fp1"], ["Fp1"])
        with self.assertRaisesRegex(ValueError, "no channel labels"):
            map_eeg_channels([], ["Fp1"])

    def test_empty_one_and_multiple_stream_selection_are_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "no LSL streams"):
            choose_stream([], 0)
        with self.assertRaisesRegex(RuntimeError, "automatic selection is disabled"):
            choose_stream(["only"], None)
        with self.assertRaisesRegex(RuntimeError, "multiple streams"):
            choose_stream(["eeg", "markers"], None)
        self.assertEqual(choose_stream(["eeg", "markers"], 1), "markers")

    def test_connection_timeout_is_propagated_without_hiding_failure(self):
        import types
        class FakeInlet:
            def __init__(self, *args, **kwargs): pass
            def open_stream(self, timeout): raise TimeoutError(f"timeout {timeout}")
        previous = sys.modules.get("pylsl")
        sys.modules["pylsl"] = types.SimpleNamespace(StreamInlet=FakeInlet)
        try:
            with self.assertRaisesRegex(TimeoutError, "timeout 0.1"):
                connect(object(), 0.1)
        finally:
            if previous is None: sys.modules.pop("pylsl", None)
            else: sys.modules["pylsl"] = previous

    def test_empty_chunks_and_bad_chunk_shapes_do_not_create_windows(self):
        buffer = RollingWindowBuffer(2, window_samples=4, step_samples=2)
        self.assertEqual(buffer.push([], []), [])
        with self.assertRaisesRegex(ValueError, "shape or timestamp"):
            buffer.push([[1.0]], [1.0])

    def test_timestamp_gap_inside_window_blocks_prediction(self):
        buffer = RollingWindowBuffer(1, window_samples=4, step_samples=2, expected_rate_hz=500)
        timestamps = [0.0, 0.002, 0.004, 0.02]
        with self.assertRaisesRegex(ValueError, "timestamp gap"):
            buffer.push([[1], [2], [3], [4]], timestamps)

    def test_exact_complete_windows_and_half_overlap(self):
        buffer = RollingWindowBuffer(2, window_samples=2000, step_samples=1000)
        samples = np.column_stack((np.arange(4000), -np.arange(4000)))
        stamps = np.arange(4000, dtype=float) / 500
        windows = []
        for start in range(0, len(samples), 137):
            windows.extend(buffer.push(samples[start:start + 137], stamps[start:start + 137]))
        self.assertEqual([item[3] for item in windows], [0, 1000, 2000])
        for window, t0, t1, _ in windows:
            self.assertEqual(window.shape, (2000, 2))
            self.assertEqual(window[0, 0], t0 * 500)
            self.assertAlmostEqual(t1 - t0, 1999 / 500)
        self.assertEqual(windows[1][0][0, 0], 1000)

    def test_sampling_rate_estimate_reports_incompatible_rate_and_gaps(self):
        regular = estimate_rate(np.arange(1000) / 500, 500, 500)
        self.assertTrue(regular["compatible"])
        slow = estimate_rate(np.arange(1000) / 250, 250, 500)
        self.assertFalse(slow["compatible"])
        values = np.arange(100, dtype=float) / 500
        values[51:] += 0.2
        self.assertGreater(estimate_rate(values, 500, 500)["irregular_or_gap_interval_count"], 0)

    def test_feature_vector_exactly_matches_existing_offline_extractor_and_saved_model_loads(self):
        config = json.loads((ROOT / "configs" / "eeg_config.json").read_text(encoding="utf-8"))
        artifact = TrainedModel(ROOT / "models" / "stress_model.pkl",
                                ROOT / "models" / "model_metadata.json",
                                ROOT / "configs" / "eeg_config.json")
        t = np.arange(2000) / 500
        data = np.column_stack([np.sin(2 * np.pi * (7 + i % 12) * t) for i in range(19)])
        offline = extract_window_features(data, artifact.channels, config)
        live_features = artifact.extract_features(data)
        self.assertEqual(list(live_features), list(offline))
        np.testing.assert_array_equal(list(live_features.values()), list(offline.values()))
        live_result = artifact.predict_window(data, 10.0, 13.998, 1)
        self.assertEqual(list(offline), artifact.feature_names)
        self.assertIn(live_result["prediction"], (0, 1))
        self.assertEqual(live_result["predicted_condition"], {0: "baseline", 1: "stress"}[live_result["prediction"]])
        self.assertIn("baseline_probability", live_result)

    def test_mocked_lsl_samples_are_recorded_without_changing_values_or_timestamps(self):
        import argparse
        from unittest.mock import patch
        from src.acquisition import cli
        class FakeInlet:
            def __init__(self): self.sent = False; self.closed = False
            def info(self): return _Info(["Fp1", "REF"])
            def pull_chunk(self, timeout, max_samples):
                if self.sent: return [], []
                self.sent = True
                return [[1.23456789, -2.5], [3.0, 4.0]], [42.123456789, 42.125456789]
            def close_stream(self): self.closed = True
        inlet = FakeInlet()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(config=ROOT / "configs" / "eeg_config.json", resolve_wait=0.1,
                stream_index=0, connect_timeout=1.0, rate_tolerance=0.02, neo4j=False,
                event_stream_index=None, duration=0.01, disconnect_timeout=1.0, output=Path(tmp),
                db_raw_seconds=0.0)
            with patch.object(cli, "resolve_streams", return_value=[_Info([])]), \
                    patch.object(cli, "connect", return_value=inlet):
                self.assertEqual(cli._run(args, infer=False), 0)
            output = next((Path(tmp) / "recordings").glob("*.csv"))
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[1], ["0", "42.123456789", "1.23456789", "-2.5"])
            self.assertEqual(rows[2], ["1", "42.125456789", "3.0", "4.0"])
            self.assertTrue(inlet.closed)

    def test_receiver_queues_only_configured_raw_seconds_to_mock_database(self):
        import argparse
        from unittest.mock import patch
        from src.acquisition import cli
        from src.storage import neo4j_store
        class FakeWriter:
            instances = []
            def __init__(self):
                self.sessions, self.chunks, self.closed = [], [], False
                self.__class__.instances.append(self)
            def submit_session(self, item): self.sessions.append(item)
            def submit_prediction(self, item): pass
            def submit_event(self, item): pass
            def submit_sample_chunk(self, item): self.chunks.append(item)
            def close(self): self.closed = True
        class FakeInlet:
            def __init__(self): self.sent = False
            def info(self): return _Info(["Fp1", "REF"])
            def pull_chunk(self, timeout, max_samples):
                if self.sent: return [], []
                self.sent = True
                values = [[float(index), float(-index)] for index in range(1000)]
                timestamps = [index / 500 for index in range(1000)]
                return values, timestamps
            def close_stream(self): pass
        FakeWriter.instances.clear()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(config=ROOT / "configs" / "eeg_config.json", resolve_wait=0.1,
                stream_index=0, connect_timeout=1.0, rate_tolerance=0.02, neo4j=True,
                event_stream_index=None, duration=0.02, disconnect_timeout=1.0, output=Path(tmp),
                db_raw_seconds=2.0)
            with patch.object(cli, "resolve_streams", return_value=[_Info([])]), \
                    patch.object(cli, "connect", return_value=FakeInlet()), \
                    patch.object(neo4j_store, "Neo4jWriter", FakeWriter):
                self.assertEqual(cli._run(args, infer=False), 0)
        fake = FakeWriter.instances[0]
        self.assertTrue(fake.closed)
        self.assertEqual([chunk["sample_count"] for chunk in fake.chunks], [500, 500])
        self.assertEqual([chunk["first_sample_index"] for chunk in fake.chunks], [0, 500])
        stored = [row for chunk in fake.chunks for row in json.loads(chunk["values_json"])]
        self.assertEqual(stored, [[float(i), float(-i)] for i in range(1000)])

    def test_neo4j_session_prediction_and_event_queries_with_mock_driver(self):
        import types
        from src.storage.neo4j_store import Neo4jWriter
        statements = []
        class Result:
            def consume(self): return None
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def run(self, query, **params):
                statements.append((query, params)); return Result()
            def execute_write(self, callback, payload): return callback(self, payload)
        class Driver:
            def verify_connectivity(self): return None
            def session(self, **kwargs): return Session()
            def close(self): return None
        fake_neo4j = types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(
            driver=lambda uri, auth: Driver()))
        previous = sys.modules.get("neo4j")
        sys.modules["neo4j"] = fake_neo4j
        try:
            writer = Neo4jWriter("bolt://fake", "user", "password")
            writer.submit_session({"session_id": "s1", "source_id": "source",
                "stream_name": "EEG", "stream_type": "EEG", "properties": {"sample_count": 2}})
            writer.submit_prediction({"session_id": "s1", "prediction_id": "s1:1",
                "properties": {"prediction": 1}})
            writer.submit_event({"session_id": "s1", "event_id": "s1:e1",
                "timestamp": 1.5, "values": ["stress_start"]})
            writer.submit_sample_chunk({"session_id": "s1", "chunk_id": "s1:0", "chunk_index": 0,
                "first_sample_index": 0, "sample_count": 1, "first_timestamp": 1.0,
                "last_timestamp": 1.0, "sampling_rate_hz": 500.0, "channel_labels": ["Fp1"],
                "values_json": "[[0.1]]", "timestamps_json": "[1.0]"})
            writer.close()
            all_queries = "\n".join(query for query, _ in statements)
            self.assertIn("EEGSession", all_queries)
            self.assertIn("EEGWindowPrediction", all_queries)
            self.assertIn("EEGEvent", all_queries)
            self.assertIn("EEGDataChunk", all_queries)
            self.assertEqual(writer.errors, [])
        finally:
            if previous is None: sys.modules.pop("neo4j", None)
            else: sys.modules["neo4j"] = previous

    def test_neo4j_reader_returns_session_prediction_and_event_records_with_mock_driver(self):
        import types
        from src.storage.neo4j_store import Neo4jReader
        expected = {"session": {"session_id": "s1"}, "stream": {"name": "test"},
                    "predictions": [{"prediction": 1}], "events": [{"values": ["marker"]}]}
        class Result:
            def __init__(self, query): self.query = query
            def data(self):
                if "c.channel_labels AS labels" in self.query:
                    return [{"labels": ["Fp1"], "values_json": "[[1.0],[2.0]]",
                             "timestamps_json": "[10.0,10.002]", "sample_count": 2,
                             "sampling_rate_hz": 500.0},
                            {"labels": ["Fp1"], "values_json": "[[3.0],[4.0]]",
                             "timestamps_json": "[10.004,10.006]", "sample_count": 2,
                             "sampling_rate_hz": 500.0}]
                return [{"session": {"session_id": "s1"}, "prediction_count": 1,
                         "raw_chunk_count": 2, "raw_sample_count": 4}]
            def single(self): return expected
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def execute_read(self, callback): return callback(self)
            def run(self, query, **params): return Result(query)
        class Driver:
            def verify_connectivity(self): return None
            def session(self, **kwargs): return Session()
            def close(self): return None
        fake_neo4j = types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(
            driver=lambda uri, auth: Driver()))
        previous = sys.modules.get("neo4j")
        sys.modules["neo4j"] = fake_neo4j
        try:
            reader = Neo4jReader("bolt://fake", "user", "password", "db")
            self.assertEqual(reader.list_sessions(2)[0]["prediction_count"], 1)
            self.assertEqual(reader.get_session("s1"), expected)
            labels, samples, timestamps, rate = reader.export_sample_chunks("s1", 3)
            self.assertEqual(labels, ["Fp1"])
            self.assertEqual(samples, [[1.0], [2.0], [3.0]])
            self.assertEqual(timestamps, [10.0, 10.002, 10.004])
            self.assertEqual(rate, 500.0)
            reader.close()
        finally:
            if previous is None: sys.modules.pop("neo4j", None)
            else: sys.modules["neo4j"] = previous

    def test_synthetic_publisher_produces_labeled_19_channel_test_data(self):
        from src.acquisition.simulated_publisher import CHANNELS, synthetic_chunk
        data = synthetic_chunk(0, 25, 500.0)
        self.assertEqual(len(CHANNELS), 19)
        self.assertEqual(data.shape, (25, 19))
        self.assertTrue(np.isfinite(data).all())

    def test_bounded_raw_chunk_collector_preserves_order_and_respects_sample_limit(self):
        from src.storage.neo4j_store import SampleChunkCollector
        class Writer:
            def __init__(self): self.chunks = []
            def submit_sample_chunk(self, item): self.chunks.append(item)
        writer = Writer()
        collector = SampleChunkCollector(writer, "s1", ["Fp1", "REF"], 500, 8, chunk_samples=4)
        values = [[index, -index] for index in range(10)]
        times = [index / 500 for index in range(10)]
        collector.add(values[:3], times[:3], 0)
        collector.add(values[3:6], times[3:6], 3)
        collector.add(values[6:], times[6:], 6)
        collector.flush()
        self.assertEqual([item["first_sample_index"] for item in writer.chunks], [0, 4])
        self.assertEqual([item["sample_count"] for item in writer.chunks], [4, 4])
        rebuilt = [row for item in writer.chunks for row in json.loads(item["values_json"])]
        self.assertEqual(rebuilt, values[:8])
        self.assertEqual(collector.samples_queued, 8)

    def test_local_recording_incrementally_writes_values_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = LocalRecording(Path(tmp), "session-test", ["Fp1", "REF"])
            rec.write_samples([[1.25, 2.5]], [123.456])
            rec.write_prediction({"window_index": 1, "window_start_timestamp": 1.0,
                "window_end_timestamp": 4.0, "prediction": 0, "predicted_condition": "baseline",
                "baseline_probability": 0.8, "stress_probability": 0.2,
                "model_algorithm": "svm_rbf", "model_version": "test"})
            rec.write_metadata({"sample_count": 1, "channel_count": 2})
            rec.close()
            with rec.csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["sample_index", "lsl_timestamp", "Fp1", "REF"])
            self.assertEqual(rows[1], ["0", "123.456", "1.25", "2.5"])
            self.assertEqual(json.loads(rec.metadata_path.read_text(encoding="utf-8"))["sample_count"], 1)
            with rec.predictions_path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["predicted_condition"], "baseline")


if __name__ == "__main__":
    unittest.main()
