"""End-to-end tests for the multi-agent research pipeline, Skeptic loop, and FakeLLM."""

from __future__ import annotations

from aetheris.agents.renderer import compute_metrics
from aetheris.core.config import Settings
from aetheris.core.models import Verdict
from aetheris.demo import DEMO_PAPERS, DemoLLM, DemoSource
from aetheris.llm.hashing import HashingEmbedder
from aetheris.orchestrator.engine import RunStatus
from aetheris.pipeline import run_research
from aetheris.tracing.store import TraceStore


async def test_full_pipeline_with_skeptic_revision_loop() -> None:
    store = TraceStore(":memory:")
    settings = Settings(max_skeptic_rounds=1, retrieval_mode="graph_hybrid")
    result = await run_research(
        "Do retrieval-augmented generation models reduce factual hallucinations?",
        settings=settings,
        llm=DemoLLM(inject_initial_overclaim=True),
        embedder=HashingEmbedder(dim=256),
        sources=(
            DemoSource(DEMO_PAPERS[:5], name="arxiv"),
            DemoSource(DEMO_PAPERS[3:], name="openalex"),
        ),
        trace_store=store,
        run_id="test-demo-run",
    )

    assert result.status is RunStatus.COMPLETED
    # Pipeline loops through skeptic -> retrieve -> synthesize -> extract_claims -> verify
    assert result.steps == [
        "plan",
        "retrieve",
        "synthesize",
        "extract_claims",
        "verify",
        "skeptic",
        "retrieve",
        "synthesize",
        "extract_claims",
        "verify",
        "skeptic",
        "render",
    ]
    assert result.state.skeptic_rounds == 2
    assert len(result.state.skeptic_history) == 2
    assert result.state.skeptic_history[0].requires_revision is True
    assert result.state.skeptic_history[1].requires_revision is False

    metrics = compute_metrics(result.state)
    assert metrics.total_claims >= 4
    assert metrics.supported == metrics.total_claims
    assert metrics.refuted == 0
    assert metrics.citation_accuracy == 1.0
    assert result.state.report_markdown is not None
    assert "## Adversarial Skeptic Audit Log" in result.state.report_markdown

    # Verify every claim's EvidenceSpan slices the exact quote from the paper abstract
    for claim in result.state.claims:
        assert claim.verdict is Verdict.SUPPORTED
        assert claim.evidence
        for span in claim.evidence:
            paper = result.state.papers[span.paper_id]
            assert paper.abstract[span.char_start : span.char_end] == span.quote

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "test-demo-run"
    events = store.get_events("test-demo-run")
    assert len(events) > 15
    store.close()
