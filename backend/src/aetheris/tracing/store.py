"""SQLite trace store.

Every run and every event is persisted so that runs can be inspected, costed
and resumed from any completed node (`load_snapshot`).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aetheris.core.events import Event, EventType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    llm_calls    INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    duration_ms  INTEGER,
    error        TEXT,
    final_state  TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    ts       TEXT NOT NULL,
    type     TEXT NOT NULL,
    node     TEXT,
    data     TEXT NOT NULL,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, seq);
"""


class TraceStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def start_run(self, run_id: str, question: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, question, status, created_at) "
                "VALUES (?, ?, 'running', ?)",
                (run_id, question, datetime.now(UTC).isoformat()),
            )

    def finish_run(self, run_id: str, final_state: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET final_state = ? WHERE run_id = ?",
                (json.dumps(final_state), run_id),
            )

    def record(self, event: Event) -> None:
        """Suitable for passing directly to ``EventBus.subscribe``."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO events (run_id, seq, ts, type, node, data, snapshot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.seq,
                    event.ts.isoformat(),
                    event.type.value,
                    event.node,
                    json.dumps(event.data),
                    json.dumps(event.snapshot) if event.snapshot is not None else None,
                ),
            )
            if event.type is EventType.RUN_FINISHED:
                d = event.data
                self._conn.execute(
                    "UPDATE runs SET status=?, finished_at=?, llm_calls=?, total_tokens=?, "
                    "duration_ms=?, error=? WHERE run_id=?",
                    (
                        d.get("status", "unknown"),
                        event.ts.isoformat(),
                        d.get("llm_calls", 0),
                        d.get("total_tokens", 0),
                        d.get("duration_ms"),
                        d.get("error"),
                        event.run_id,
                    ),
                )

    def list_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, question, status, created_at, llm_calls, total_tokens, "
            "duration_ms, error FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_events(self, run_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [
            Event(
                run_id=r["run_id"],
                seq=r["seq"],
                ts=datetime.fromisoformat(r["ts"]),
                type=EventType(r["type"]),
                node=r["node"],
                data=json.loads(r["data"]),
                snapshot=json.loads(r["snapshot"]) if r["snapshot"] else None,
            )
            for r in rows
        ]

    def load_snapshot(self, run_id: str, node: str) -> dict[str, Any] | None:
        """Latest state snapshot produced by ``node`` in ``run_id``."""
        row = self._conn.execute(
            "SELECT snapshot FROM events WHERE run_id=? AND node=? AND type='node_completed' "
            "AND snapshot IS NOT NULL ORDER BY seq DESC LIMIT 1",
            (run_id, node),
        ).fetchone()
        return json.loads(row["snapshot"]) if row else None

