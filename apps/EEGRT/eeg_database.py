"""Optional asynchronous Neo4j sink for recording sessions and predictions.

High-rate EEG remains in local CSV; the graph stores session metadata, file
references, and predictions. Neo4j connectivity is optional and lazy-loaded.
"""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any


def load_project_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from this application's .env."""
    env_path = path or Path(__file__).resolve().parent / ".env"
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


class Neo4jWriter:
    def __init__(self, uri: str | None = None, username: str | None = None,
                 password: str | None = None, database: str | None = None,
                 queue_size: int = 5000):
        load_project_env()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j driver missing; install this folder's requirements.txt") from exc
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
                "CREATE CONSTRAINT eeg_participant_id IF NOT EXISTS FOR (n:EEGParticipant) REQUIRE n.participant_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_session_id IF NOT EXISTS FOR (n:EEGSession) REQUIRE n.session_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_stream_id IF NOT EXISTS FOR (n:EEGStream) REQUIRE n.source_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_prediction_id IF NOT EXISTS FOR (n:EEGWindowPrediction) REQUIRE n.prediction_id IS UNIQUE",
                "CREATE CONSTRAINT eeg_event_id IF NOT EXISTS FOR (n:EEGEvent) REQUIRE n.event_id IS UNIQUE",
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
            WITH s
            FOREACH (_ IN CASE WHEN $participant_id IS NULL OR $participant_id = '' THEN [] ELSE [1] END |
                MERGE (p:EEGParticipant {participant_id: $participant_id})
                SET p.display_name = $participant_name
                MERGE (s)-[:FOR_PARTICIPANT]->(p)
            )
        """, session_id=item["session_id"], properties=item["properties"],
             source_id=item.get("source_id", "unknown"), stream_name=item.get("stream_name", "unknown"),
             stream_type=item.get("stream_type", "unknown"),
             participant_id=item["properties"].get("participant_id"),
             participant_name=item["properties"].get("participant_name")).consume()

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
            raise RuntimeError("Neo4j driver missing; install this folder's requirements.txt") from exc
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
            RETURN properties(s) AS session, count(DISTINCT p) AS prediction_count
            ORDER BY session.started_at_utc DESC
            LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(lambda tx: list(tx.run(query, limit=limit).data()))
        return [{"session": dict(row["session"]), "prediction_count": row["prediction_count"]}
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

    def get_session_report(self, session_id: str) -> dict[str, Any] | None:
        """Read stored session metadata and predictions without fetching EEG samples."""
        query = """
            MATCH (s:EEGSession {session_id: $session_id})
            OPTIONAL MATCH (s)-[:HAS_PREDICTION]->(p:EEGWindowPrediction)
            WITH s, p ORDER BY p.window_index
            WITH s, collect(p) AS predictions
            RETURN properties(s) AS session,
                   [item IN predictions WHERE item IS NOT NULL | properties(item)] AS predictions
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_read(lambda tx: tx.run(query, session_id=session_id).single())
        if row is None:
            return None
        return {"session": dict(row["session"]),
                "predictions": [dict(item) for item in row["predictions"]]}

    def close(self) -> None:
        self.driver.close()

