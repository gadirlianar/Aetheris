"""Async graph executor with retries, timeouts, budgets, tracing and resumption."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel

from aetheris.core.budget import Budget
from aetheris.core.errors import BudgetExceededError
from aetheris.core.events import EventBus, EventType, current_node
from aetheris.orchestrator.graph import END, Graph, NodeSpec


class RunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS = "max_steps"


@dataclass
class RunResult[S]:
    state: S
    status: RunStatus
    steps: list[str] = field(default_factory=list)
    error: str | None = None
    failed_node: str | None = None


Sleep = Callable[[float], Awaitable[None]]


class Engine:
    """Executes a :class:`Graph` one node at a time.

    Guarantees:
      * a node only ever sees a state produced by a *successful* predecessor;
      * transient failures are retried per the node's :class:`RetryPolicy`;
      * the run never raises: the outcome is reported via :class:`RunResult`,
        carrying the last good state so partial work is never lost;
      * after every node a full state snapshot is emitted, so any run can be
        resumed from any node with ``start_at``.
    """

    def __init__(
        self,
        events: EventBus,
        *,
        budget: Budget | None = None,
        max_steps: int = 50,
        snapshots: bool = True,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._events = events
        self._budget = budget or Budget()
        self._max_steps = max_steps
        self._snapshots = snapshots
        self._sleep = sleep

    async def run[S: BaseModel, C](
        self, graph: Graph[S, C], state: S, ctx: C, *, start_at: str | None = None
    ) -> RunResult[S]:
        graph.validate()
        current = start_at or graph.entry
        assert current is not None
        if current not in graph.nodes:
            raise ValueError(f"unknown start node '{current}'")

        result = RunResult(state=state, status=RunStatus.COMPLETED)
        await self._events.emit(EventType.RUN_STARTED, entry=current)
        started = time.perf_counter()

        while current != END:
            if len(result.steps) >= self._max_steps:
                result.status = RunStatus.MAX_STEPS
                result.error = f"exceeded {self._max_steps} steps"
                break
            spec = graph.nodes[current]
            try:
                self._budget.check()
                result.state = await self._run_node(spec, result.state, ctx)
                result.steps.append(current)
                current = graph.next_node(current, result.state)
            except BudgetExceededError as exc:
                result.status, result.error, result.failed_node = (
                    RunStatus.BUDGET_EXCEEDED,
                    str(exc),
                    current,
                )
                await self._events.emit(EventType.NODE_FAILED, node=current, error=str(exc))
                break
            except Exception as exc:
                result.status, result.error, result.failed_node = (
                    RunStatus.FAILED,
                    f"{type(exc).__name__}: {exc}",
                    current,
                )
                await self._events.emit(EventType.NODE_FAILED, node=current, error=result.error)
                break

        await self._events.emit(
            EventType.RUN_FINISHED,
            status=result.status.value,
            steps=result.steps,
            error=result.error,
            duration_ms=round((time.perf_counter() - started) * 1000),
            llm_calls=self._budget.llm_calls,
            total_tokens=self._budget.total_tokens,
        )
        return result

    async def _run_node[S: BaseModel, C](self, spec: NodeSpec[S, C], state: S, ctx: C) -> S:
        token = current_node.set(spec.name)
        try:
            attempt = 1
            while True:
                await self._events.emit(EventType.NODE_STARTED, attempt=attempt)
                started = time.perf_counter()
                try:
                    async with asyncio.timeout(spec.timeout_s):
                        new_state = await spec.node.run(state, ctx)
                except spec.retry.retry_on as exc:
                    if attempt >= spec.retry.max_attempts:
                        raise
                    delay = spec.retry.delay(attempt)
                    await self._events.emit(
                        EventType.NODE_RETRY,
                        attempt=attempt,
                        delay_s=round(delay, 3),
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    )
                    await self._sleep(delay)
                    attempt += 1
                    continue
                await self._events.emit(
                    EventType.NODE_COMPLETED,
                    attempt=attempt,
                    duration_ms=round((time.perf_counter() - started) * 1000),
                    snapshot=new_state.model_dump(mode="json") if self._snapshots else None,
                )
                return new_state
        finally:
            current_node.reset(token)
