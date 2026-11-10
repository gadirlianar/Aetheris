"""SQLite trace and checkpoint store hardened for concurrent access and deterministic replay.

Hardening guarantees:
  * Enables ``PRAGMA journal_mode=WAL;``, ``PRAGMA synchronous=NORMAL;``, and
    ``PRAGMA busy_timeout=5000;`` so concurrent readers (SSE stream, CLI inspector)
    and writers never block one another on disk.
  * Guards all connection operations with a re-entrant lock and automatic exponential
    backoff retry on ``sqlite3.OperationalError`` (``SQLITE_BUSY`` / ``database is locked``).
  * Supports deterministic state resumption and fork/replay from any graph node via
    :meth:`load_state_before_node` and :meth:`load_snapshot`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from aetheris.core.events import Event, EventType

_T = TypeVar("_T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    llm_calls    INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd     REAL DEFAULT 0.0,
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
    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        busy_timeout_ms: int = 5000,
        max_busy_retries: int = 5,
    ) -> None:
        self._path = str(path)
        self._max_retries = max_busy_retries
        self._lock = threading.RLock()
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._path,
            timeout=busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._ensure_columns()

    def _ensure_columns(self) -> None:
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(runs);").fetchall()}
        if "cost_usd" not in cols:
            self._conn.execute("ALTER TABLE runs ADD COLUMN cost_usd REAL DEFAULT 0.0;")

    def _with_retry(self, fn: Callable[[], _T]) -> _T:
        delay = 0.02
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._lock:
                    return fn()
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt < self._max_retries:
                    time.sleep(delay)
                    delay = min(0.5, delay * 2.0)
                    continue
                raise
        raise RuntimeError("unreachable")

    @property
    def journal_mode(self) -> str:
        def _query() -> str:
            row = self._conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else "unknown"

        return self._with_retry(_query)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def start_run(self, run_id: str, question: str) -> None:
        def _op() -> None:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO runs (run_id, question, status, created_at) "
                    "VALUES (?, ?, 'running', ?)",
                    (run_id, question, datetime.now(UTC).isoformat()),
                )

        self._with_retry(_op)

    def finish_run(self, run_id: str, final_state: dict[str, Any]) -> None:
        def _op() -> None:
            cost = 0.0
            telemetry = final_state.get("cost_telemetry")
            if isinstance(telemetry, dict):
                raw_c = telemetry.get("total_cost_usd", 0.0)
                if isinstance(raw_c, int | float):
                    cost = float(raw_c)
            with self._conn:
                self._conn.execute(
                    "UPDATE runs SET final_state = ?, cost_usd = ? WHERE run_id = ?",
                    (json.dumps(final_state), cost, run_id),
                )

        self._with_retry(_op)

    def record(self, event: Event) -> None:
        """Suitable for passing directly to ``EventBus.subscribe``."""

        def _op() -> None:
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
                        "cost_usd=?, duration_ms=?, error=? WHERE run_id=?",
                        (
                            d.get("status", "unknown"),
                            event.ts.isoformat(),
                            d.get("llm_calls", 0),
                            d.get("total_tokens", 0),
                            d.get("cost_usd", 0.0),
                            d.get("duration_ms"),
                            d.get("error"),
                            event.run_id,
                        ),
                    )

        self._with_retry(_op)

    def list_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        def _op() -> list[dict[str, Any]]:
            rows = self._conn.execute(
                "SELECT run_id, question, status, created_at, llm_calls, total_tokens, "
                "cost_usd, duration_ms, error FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        return self._with_retry(_op)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        def _op() -> dict[str, Any] | None:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

        return self._with_retry(_op)

    def get_events(self, run_id: str) -> list[Event]:
        def _op() -> list[Event]:
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

        return self._with_retry(_op)

    def load_snapshot(self, run_id: str, node: str) -> dict[str, Any] | None:
        """Latest state snapshot produced by ``node`` in ``run_id``."""

        def _op() -> dict[str, Any] | None:
            row = self._conn.execute(
                "SELECT snapshot FROM events WHERE run_id=? AND node=? AND type='node_completed' "
                "AND snapshot IS NOT NULL ORDER BY seq DESC LIMIT 1",
                (run_id, node),
            ).fetchone()
            return json.loads(row["snapshot"]) if row else None

        return self._with_retry(_op)

    def load_state_before_node(
        self, run_id: str, from_node: str, pipeline_order: Sequence[str]
    ) -> dict[str, Any] | None:
        """Load the checkpoint snapshot immediately preceding ``from_node`` in ``run_id``."""

        def _op() -> dict[str, Any] | None:
            run_row = self._conn.execute(
                "SELECT question, final_state FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return None

            # Find the first/latest occurrence of `from_node` in completed events and take the
            # snapshot from the immediately preceding `node_completed` event.
            rows = self._conn.execute(
                "SELECT seq, node, snapshot FROM events "
                "WHERE run_id=? AND type='node_completed' AND snapshot IS NOT NULL "
                "ORDER BY seq ASC",
                (run_id,),
            ).fetchall()

            prev_snapshot: str | None = None
            for r in rows:
                if r["node"] == from_node:
                    if prev_snapshot is not None:
                        return dict(json.loads(prev_snapshot))
                    return {"run_id": run_id, "question": str(run_row["question"])}
                prev_snapshot = r["snapshot"]

            # Fallback: look up predecessor in canonical pipeline order
            if from_node in pipeline_order:
                idx = list(pipeline_order).index(from_node)
                if idx == 0:
                    return {"run_id": run_id, "question": str(run_row["question"])}
                for pred in reversed(pipeline_order[:idx]):
                    snap = self.load_snapshot(run_id, pred)
                    if snap is not None:
                        return snap

            if prev_snapshot is not None:
                return dict(json.loads(prev_snapshot))
            if run_row["final_state"]:
                return dict(json.loads(run_row["final_state"]))
            return {"run_id": run_id, "question": str(run_row["question"])}

        return self._with_retry(_op)
