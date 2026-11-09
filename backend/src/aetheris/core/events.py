"""Structured run events.

Every observable thing that happens during a run is an ``Event``. The same
stream feeds the CLI live view, the SQLite trace store and (in a later phase)
the Server-Sent-Events endpoint behind the dashboard.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# The node currently executing. Set by the engine so that events emitted deep
# inside an agent (LLM calls, source queries) are attributed automatically.
current_node: ContextVar[str | None] = ContextVar("aetheris_current_node", default=None)


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_RETRY = "node_retry"
    LLM_CALL = "llm_call"
    EMBEDDING_CALL = "embedding_call"
    SOURCE_QUERY = "source_query"
    PROGRESS = "progress"


class Event(BaseModel):
    run_id: str
    seq: int
    ts: datetime
    type: EventType
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Full state snapshot (node_completed only). Kept out of `data` so that "
        "live consumers can drop it cheaply.",
    )


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Minimal in-process pub/sub. Handlers run in subscription order."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._seq = 0
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    async def emit(
        self,
        type: EventType,
        *,
        node: str | None = None,
        snapshot: dict[str, Any] | None = None,
        **data: Any,
    ) -> Event:
        self._seq += 1
        event = Event(
            run_id=self.run_id,
            seq=self._seq,
            ts=datetime.now(UTC),
            type=type,
            node=node if node is not None else current_node.get(),
            data=data,
            snapshot=snapshot,
        )
        for handler in self._handlers:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        return event
