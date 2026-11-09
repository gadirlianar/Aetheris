"""Unit tests for core domain models, budget enforcement, and event bus."""

from __future__ import annotations

import pytest

from aetheris.core.budget import Budget
from aetheris.core.errors import BudgetExceededError
from aetheris.core.events import Event, EventBus, EventType, current_node
from aetheris.core.models import (
    Claim,
    DraftReport,
    Paper,
    ReportSection,
    ReportSentence,
    ResearchPlan,
    ResearchState,
    SubQuestion,
    Verdict,
)


def test_paper_short_citation() -> None:
    p_none = Paper(
        id="arxiv:1",
        source="arxiv",
        title="T",
        abstract="A",
        authors=(),
        year=2024,
        url="https://arxiv.org/abs/1",
    )
    assert p_none.short_citation() == "Unknown, 2024"

    p_single = p_none.model_copy(update={"authors": ("Ada Lovelace",), "year": None})
    assert p_single.short_citation() == "Lovelace"

    p_multi = p_none.model_copy(update={"authors": ("Patrick Lewis", "Douwe Kiela")})
    assert p_multi.short_citation() == "Lewis et al., 2024"


def test_research_plan_deduplicates_queries() -> None:
    plan = ResearchPlan(
        objective="Obj",
        sub_questions=[
            SubQuestion(question="Q1", rationale="R1", search_queries=[" rag ", "bm25", "rag"]),
            SubQuestion(question="Q2", rationale="R2", search_queries=["bm25", "graphrag", " "]),
        ],
    )
    assert plan.all_queries() == ["rag", "bm25", "graphrag"]


def test_draft_and_state_lookups() -> None:
    s1 = ReportSentence(id="s1", text="First.")
    s2 = ReportSentence(id="s2", text="Second.")
    draft = DraftReport(
        title="Title",
        sections=[ReportSection(heading="H1", sentences=[s1, s2])],
    )
    assert [s.id for s in draft.iter_sentences()] == ["s1", "s2"]
    assert draft.sentence("s2") == s2
    assert draft.sentence("s99") is None

    state = ResearchState(
        run_id="r1",
        question="Q",
        draft=draft,
        claims=[
            Claim(id="c1", sentence_id="s1", text="C1", verdict=Verdict.SUPPORTED),
            Claim(id="c2", sentence_id="s2", text="C2", verdict=Verdict.REFUTED),
        ],
    )
    assert [c.id for c in state.claims_for("s1")] == ["c1"]


def test_budget_limits() -> None:
    now = 100.0
    budget = Budget(max_llm_calls=2, max_total_tokens=100, max_seconds=10.0, clock=lambda: now)
    budget.check()
    budget.charge(40)
    budget.check()
    budget.charge(40)
    with pytest.raises(BudgetExceededError, match="LLM call limit"):
        budget.check()

    token_budget = Budget(max_llm_calls=10, max_total_tokens=50, max_seconds=10.0)
    token_budget.charge(60)
    with pytest.raises(BudgetExceededError, match="token limit"):
        token_budget.check()

    time_budget = Budget(max_llm_calls=10, max_total_tokens=500, max_seconds=5.0, clock=lambda: now)
    now = 106.0
    with pytest.raises(BudgetExceededError, match="time limit"):
        time_budget.check()


async def test_event_bus_and_contextvar() -> None:
    bus = EventBus("run-42")
    received: list[Event] = []
    bus.subscribe(received.append)

    token = current_node.set("verify")
    try:
        ev = await bus.emit(EventType.PROGRESS, message="checking claims")
    finally:
        current_node.reset(token)

    assert ev.seq == 1
    assert ev.node == "verify"
    assert received[0].data["message"] == "checking claims"
