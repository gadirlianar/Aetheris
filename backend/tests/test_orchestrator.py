"""Unit tests for the declarative Graph and async Engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aetheris.core.budget import Budget
from aetheris.core.errors import GraphValidationError, TransientError
from aetheris.core.events import EventBus
from aetheris.orchestrator.engine import Engine, RunStatus
from aetheris.orchestrator.graph import END, Graph, RetryPolicy
from aetheris.tracing.store import TraceStore


class CounterState(BaseModel):
    count: int = 0
    flaky_attempts: int = 0


class AddNode:
    def __init__(self, delta: int = 1) -> None:
        self._delta = delta

    async def run(self, state: CounterState, ctx: None) -> CounterState:
        del ctx
        return state.model_copy(update={"count": state.count + self._delta})


class FlakyNode:
    def __init__(self, fail_times: int = 2) -> None:
        self._fail_times = fail_times

    async def run(self, state: CounterState, ctx: None) -> CounterState:
        del ctx
        attempts = state.flaky_attempts + 1
        if attempts <= self._fail_times:
            state.flaky_attempts = attempts
            raise TransientError("temporary upstream rate limit")
        return state.model_copy(update={"count": state.count + 10, "flaky_attempts": attempts})


def test_graph_validation_catches_wiring_errors() -> None:
    empty: Graph[CounterState, None] = Graph()
    with pytest.raises(GraphValidationError, match="no nodes"):
        empty.validate()

    g: Graph[CounterState, None] = Graph()
    with pytest.raises(GraphValidationError, match="reserved"):
        g.add_node(END, AddNode())

    g.add_node("a", AddNode())
    with pytest.raises(GraphValidationError, match="duplicate"):
        g.add_node("a", AddNode())

    with pytest.raises(GraphValidationError, match="no outgoing"):
        g.validate()

    g.add_edge("a", "b")
    with pytest.raises(GraphValidationError, match="already has an outgoing"):
        g.add_edge("a", END)

    with pytest.raises(GraphValidationError, match="unknown"):
        g.validate()

    g2: Graph[CounterState, None] = (
        Graph[CounterState, None]()
        .add_node("a", AddNode())
        .add_node("orphan", AddNode())
        .add_edge("a", END)
        .add_edge("orphan", END)
    )
    with pytest.raises(GraphValidationError, match="unreachable"):
        g2.validate()


async def test_engine_conditional_loop_retry_and_resumption() -> None:
    store = TraceStore(":memory:")
    bus = EventBus("r-loop")
    bus.subscribe(store.record)

    policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
    graph = (
        Graph[CounterState, None]()
        .add_node("inc", AddNode(1))
        .add_node("flaky", FlakyNode(fail_times=2), retry=policy)
        .add_conditional("inc", lambda s: "inc" if s.count < 3 else "flaky", ("inc", "flaky"))
        .add_edge("flaky", END)
    )
    assert "flowchart TD" in graph.to_mermaid()

    async def no_sleep(_s: float) -> None:
        return None

    engine = Engine(bus, sleep=no_sleep)
    result = await engine.run(graph, CounterState(), None)
    assert result.status is RunStatus.COMPLETED
    assert result.state.count == 13
    assert result.steps == ["inc", "inc", "inc", "flaky"]

    # Resume from the snapshot after 'inc'
    snap = store.load_snapshot("r-loop", "inc")
    assert snap is not None
    resumed_state = CounterState.model_validate(snap)
    resumed = await engine.run(graph, resumed_state, None, start_at="flaky")
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.state.count == 13
    store.close()


async def test_engine_max_steps_and_budget_exceeded() -> None:
    graph = (
        Graph[CounterState, None]()
        .add_node("loop", AddNode(1))
        .add_conditional("loop", lambda _s: "loop", ("loop", END))
    )
    res = await Engine(EventBus("r-max"), max_steps=4).run(graph, CounterState(), None)
    assert res.status is RunStatus.MAX_STEPS
    assert len(res.steps) == 4

    exhausted = Budget(max_llm_calls=0)
    res_budget = await Engine(EventBus("r-b"), budget=exhausted).run(
        Graph[CounterState, None]().add_node("a", AddNode()).add_edge("a", END),
        CounterState(),
        None,
    )
    assert res_budget.status is RunStatus.BUDGET_EXCEEDED
