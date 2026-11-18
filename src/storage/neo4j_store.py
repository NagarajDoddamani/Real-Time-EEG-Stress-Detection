"""Optional asynchronous Neo4j sink for recording sessions and predictions.

High-rate EEG remains in local CSV; the graph stores session metadata, file
references, and predictions. Neo4j connectivity is optional and lazy-loaded.
"""
from __future__ import annotations

import os
import json
import queue
import threading
from pathlib import Path
from typing import Any


def load_project_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from project .env without overriding shell vars."""
    env_path = path or Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


class SampleChunkCollector:
    """Buffer source samples into bounded, graph-friendly chunks."""
    def __init__(self, writer: "Neo4jWriter", session_id: str, channel_labels: list[str],
                 sampling_rate_hz: float, max_samples: int, chunk_samples: int = 500):
        if sampling_rate_hz <= 0 or max_samples < 1 or chunk_samples < 1:
            raise ValueError("invalid raw EEG graph-storage limits")
        self.writer, self.session_id = writer, session_id
        self.channel_labels, self.sampling_rate_hz = list(channel_labels), float(sampling_rate_hz)
        self.max_samples, self.chunk_samples = int(max_samples), int(chunk_samples)
        self.samples: list[list[float]] = []
        self.timestamps: list[float] = []
        self.chunk_index = 0
        self.samples_queued = 0

    def add(self, samples: list[list[float]], timestamps: list[float], first_sample_index: int) -> None:
        if len(samples) != len(timestamps):
            raise ValueError("raw sample/timestamp count mismatch for graph storage")
        allowed = min(len(samples), self.max_samples - self.samples_queued - len(self.samples))
        for offset, (row, timestamp) in enumerate(zip(samples[:allowed], timestamps[:allowed])):
            if len(row) != len(self.channel_labels):
                raise ValueError("raw EEG row width differs from stored channel labels")
            self.samples.append([float(value) for value in row])
            self.timestamps.append(float(timestamp))
            if len(self.samples) == self.chunk_samples:
                self._queue_chunk(first_sample_index + offset + 1 - self.chunk_samples)

    def _queue_chunk(self, start_sample_index: int) -> None:
        if not self.samples:
            return
        chunk = {"session_id": self.session_id,
            "chunk_id": f"{self.session_id}:{self.chunk_index}", "chunk_index": self.chunk_index,
            "first_sample_index": int(start_sample_index), "sample_count": len(self.samples),
            "first_timestamp": self.timestamps[0], "last_timestamp": self.timestamps[-1],
            "sampling_rate_hz": self.sampling_rate_hz, "channel_labels": self.channel_labels,
            "values_json": json.dumps(self.samples, separators=(",", ":")),
            "timestamps_json": json.dumps(self.timestamps, separators=(",", ":"))}
        self.writer.submit_sample_chunk(chunk)
        self.samples_queued += len(self.samples)
        self.chunk_index += 1
        self.samples.clear(); self.timestamps.clear()

    def flush(self, first_sample_index: int = 0) -> None:
        if self.samples:
            self._queue_chunk(first_sample_index)


class Neo4jWriter:
    def __init__(self, uri: str | None = None, username: str | None = None,
                 password: str | None = None, database: str | None = None,
                 queue_size: int = 5000):
        load_project_env()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j driver missing; install requirements.txt") from exc
        uri = uri or os.getenv("NEO4J_URI")
        username = username or os.getenv("NEO4J_USERNAME")
        password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE") or "neo4j"
        if not uri or not username or not password:
            raise RuntimeError("set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        with self.driver.session(database=self.database) as session:
            for statement in (
                "CREATE CONSTRAINT eeg_session_id IF NOT EXISTS FOR (n:EEGSession) REQUIRE n.session_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_stream_id IF NOT EXISTS FOR (n:EEGStream) REQUIRE n.source_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_prediction_id IF NOT EXISTS FOR (n:EEGWindowPrediction) REQUIRE n.prediction_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_event_id IF NOT EXISTS FOR (n:EEGEvent) REQUIRE n.event_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_data_chunk_id IF NOT EXISTS FOR (n:EEGDataChunk) REQUIRE n.chunk_id IS UNIQUE",
            ):
                session.run(statement).consume()
        self.queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=queue_size)
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, name="neo4j-writer", daemon=True)
        self.thread.start()

    def submit_session(self, session: dict[str, Any]) -> None:
        self.queue.put_nowait(("session", session))

    def submit_prediction(self, prediction: dict[str, Any]) -> None:
        self.queue.put_nowait(("prediction", prediction))

    def submit_event(self, event: dict[str, Any]) -> None:
        self.queue.put_nowait(("event", event))

    def submit_sample_chunk(self, chunk: dict[str, Any]) -> None:
        """Queue one compact JSON chunk of original samples/timestamps for bounded tests."""
        self.queue.put_nowait(("samples", chunk))

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                with self.driver.session(database=self.database) as session:
                    if kind == "session":
                        session.execute_write(self._write_session, payload)
                    elif kind == "prediction":
                        session.execute_write(self._write_prediction, payload)
                    elif kind == "samples":
                        session.execute_write(self._write_sample_chunk, payload)
                    else:
                        session.execute_write(self._write_event, payload)
            except Exception as exc:  # keep acquisition thread independent
                self.errors.append(str(exc))
            finally:
                self.queue.task_done()

    @staticmethod
    def _write_session(tx: Any, item: dict[str, Any]) -> None:
        tx.run("""
            MERGE (s:EEGSession {session_id: $session_id})
            SET s += $properties
            MERGE (d:EEGStream {source_id: $source_id})
            SET d.name = $stream_name, d.type = $stream_type
            MERGE (s)-[:RECEIVED_FROM]->(d)
        """, session_id=item["session_id"], properties=item["properties"],
             source_id=item.get("source_id", "unknown"), stream_name=item.get("stream_name", "unknown"),
             stream_type=item.get("stream_type", "unknown")).consume()

    @staticmethod
    def _write_prediction(tx: Any, item: dict[str, Any]) -> None:
        tx.run("""
            MATCH (s:EEGSession {session_id: $session_id})
            MERGE (p:EEGWindowPrediction {prediction_id: $prediction_id})
            SET p += $properties
            MERGE (s)-[:HAS_PREDICTION]->(p)
        """, session_id=item["session_id"], prediction_id=item["prediction_id"],
             properties=item["properties"]).consume()

    @staticmethod
    def _write_event(tx: Any, item: dict[str, Any]) -> None:
        tx.run("""
            MATCH (s:EEGSession {session_id: $session_id})
            MERGE (e:EEGEvent {event_id: $event_id})
            SET e.timestamp = $timestamp, e.values = $values
            MERGE (s)-[:HAS_EVENT]->(e)
        """, session_id=item["session_id"], event_id=item["event_id"],
             timestamp=item["timestamp"], values=item["values"]).consume()

    @staticmethod
    def _write_sample_chunk(tx: Any, item: dict[str, Any]) -> None:
        tx.run("""
            MATCH (s:EEGSession {session_id: $session_id})
            MERGE (c:EEGDataChunk {chunk_id: $chunk_id})
            SET c.session_id = $session_id, c.chunk_index = $chunk_index,
                c.first_sample_index = $first_sample_index, c.sample_count = $sample_count,
                c.first_timestamp = $first_timestamp, c.last_timestamp = $last_timestamp,
                c.sampling_rate_hz = $sampling_rate_hz, c.channel_labels = $channel_labels,
                c.values_json = $values_json, c.timestamps_json = $timestamps_json,
                c.storage_kind = 'raw_eeg_sample_chunk'
            MERGE (s)-[:HAS_EEG_DATA_CHUNK]->(c)
        """, **item).consume()

    def close(self, timeout: float = 10.0) -> None:
        self.queue.join()
        self.queue.put(None)
        self.thread.join(timeout)
        self.driver.close()


class Neo4jReader:
    """Read session metadata and its prediction/event records from Neo4j."""
    def __init__(self, uri: str | None = None, username: str | None = None,
                 password: str | None = None, database: str | None = None):
        load_project_env()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j driver missing; install requirements.txt") from exc
        uri = uri or os.getenv("NEO4J_URI")
        username = username or os.getenv("NEO4J_USERNAME")
        password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE") or "neo4j"
        if not uri or not username or not password:
            raise RuntimeError("set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        query = """
            MATCH (s:EEGSession)
            OPTIONAL MATCH (s)-[:HAS_PREDICTION]->(p:EEGWindowPrediction)
            WITH s, count(DISTINCT p) AS prediction_count
            OPTIONAL MATCH (s)-[:HAS_EEG_DATA_CHUNK]->(c:EEGDataChunk)
            RETURN properties(s) AS session, prediction_count,
                   count(c) AS raw_chunk_count, coalesce(sum(c.sample_count), 0) AS raw_sample_count
            ORDER BY session.started_at_utc DESC
            LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(lambda tx: list(tx.run(query, limit=limit).data()))
        return [{"session": dict(row["session"]), "prediction_count": row["prediction_count"],
                 "raw_chunk_count": row["raw_chunk_count"], "raw_sample_count": row["raw_sample_count"]}
                for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        query = """
            MATCH (s:EEGSession {session_id: $session_id})
            OPTIONAL MATCH (s)-[:RECEIVED_FROM]->(stream:EEGStream)
            OPTIONAL MATCH (s)-[:HAS_PREDICTION]->(p:EEGWindowPrediction)
            WITH s, stream, collect(p) AS predictions
            OPTIONAL MATCH (s)-[:HAS_EVENT]->(e:EEGEvent)
            RETURN properties(s) AS session, properties(stream) AS stream,
                   [item IN predictions WHERE item IS NOT NULL | properties(item)] AS predictions,
                   collect(e) AS events
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_read(lambda tx: tx.run(query, session_id=session_id).single())
        if row is None:
            return None
        return {"session": dict(row["session"]),
                "stream": dict(row["stream"]) if row["stream"] else None,
                "predictions": [dict(item) for item in row["predictions"]],
                "events": [dict(item) for item in row["events"] if item is not None]}

    def get_session_properties(self, session_id: str) -> dict[str, Any] | None:
        """Read only session properties, including participant and active status."""
        query = "MATCH (s:EEGSession {session_id: $session_id}) RETURN properties(s) AS session"
        with self.driver.session(database=self.database) as session:
            row = session.execute_read(lambda tx: tx.run(query, session_id=session_id).single())
        return dict(row["session"]) if row else None

    def read_sample_chunks_after(self, session_id: str, first_chunk_index: int,
                                 limit: int = 10) -> list[dict[str, Any]]:
        """Fetch newly committed chunks in source order for live polling."""
        if first_chunk_index < 0 or not 1 <= limit <= 100:
            raise ValueError("first_chunk_index must be nonnegative and limit must be 1..100")
        query = """
            MATCH (:EEGSession {session_id: $session_id})-[:HAS_EEG_DATA_CHUNK]->(c:EEGDataChunk)
            WHERE c.chunk_index >= $first_chunk_index
            RETURN c.chunk_index AS chunk_index, c.first_sample_index AS first_sample_index,
                   c.sample_count AS sample_count, c.sampling_rate_hz AS sampling_rate_hz,
                   c.channel_labels AS channel_labels, c.values_json AS values_json,
                   c.timestamps_json AS timestamps_json
            ORDER BY c.chunk_index
            LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(lambda tx: list(tx.run(
                query, session_id=session_id, first_chunk_index=first_chunk_index, limit=limit).data()))
        chunks = []
        for row in rows:
            values = json.loads(row["values_json"])
            timestamps = json.loads(row["timestamps_json"])
            count = int(row["sample_count"])
            if len(values) != count or len(timestamps) != count:
                raise ValueError(f"database chunk {row['chunk_index']} has inconsistent sample/timestamp counts")
            chunks.append({
                "chunk_index": int(row["chunk_index"]),
                "first_sample_index": int(row["first_sample_index"]),
                "sample_count": count,
                "sampling_rate_hz": float(row["sampling_rate_hz"]),
                "channel_labels": list(row["channel_labels"]),
                "samples": values,
                "timestamps": timestamps,
            })
        return chunks

    def export_sample_chunks(self, session_id: str, max_samples: int
                             ) -> tuple[list[str], list[list[Any]], list[float], float]:
        """Read chunked raw samples in source order, capped to max_samples."""
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        query = """
            MATCH (:EEGSession {session_id: $session_id})-[:HAS_EEG_DATA_CHUNK]->(c:EEGDataChunk)
            RETURN c.channel_labels AS labels, c.values_json AS values_json,
                   c.timestamps_json AS timestamps_json, c.sample_count AS sample_count,
                   c.sampling_rate_hz AS sampling_rate_hz
            ORDER BY c.chunk_index
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(lambda tx: list(tx.run(query, session_id=session_id).data()))
        if not rows:
            raise ValueError(f"no stored raw EEG chunks found for session {session_id}")
        labels = list(rows[0]["labels"])
        samples: list[list[Any]] = []
        timestamps: list[float] = []
        for row in rows:
            if list(row["labels"]) != labels:
                raise ValueError("channel order changes between stored EEG chunks")
            chunk_samples = json.loads(row["values_json"])
            chunk_timestamps = json.loads(row["timestamps_json"])
            if len(chunk_samples) != int(row["sample_count"]) or len(chunk_timestamps) != len(chunk_samples):
                raise ValueError("stored EEG chunk count is inconsistent")
            take = min(len(chunk_samples), max_samples - len(samples))
            samples.extend(chunk_samples[:take])
            timestamps.extend(chunk_timestamps[:take])
            if len(samples) >= max_samples:
                break
        return labels, samples, timestamps, float(rows[0]["sampling_rate_hz"])

    def close(self) -> None:
        self.driver.close()

