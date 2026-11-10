"""End-to-end tests for the research pipeline, Skeptic loop, Contradiction, and Replay."""

from __future__ import annotations

from pathlib import Path

from aetheris.agents.renderer import compute_metrics
from aetheris.core.config import Settings
from aetheris.core.models import Verdict
from aetheris.demo import DEMO_PAPERS, DemoLLM, DemoSource
from aetheris.llm.hashing import HashingEmbedder
from aetheris.orchestrator.engine import RunStatus
from aetheris.pipeline import replay_research, run_research
from aetheris.tracing.store import TraceStore


async def test_full_pipeline_with_skeptic_revision_loop_and_contradiction_synthesizer(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "trace_wal.db"
    store = TraceStore(db_file)
    settings = Settings(
        max_skeptic_rounds=1,
        retrieval_mode="graph_hybrid",
        max_concurrent_claims=5,
        trace_db=db_file,
    )
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
    # -> skeptic -> contradiction -> render
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
        "contradiction",
        "render",
    ]
    assert result.state.skeptic_rounds == 2
    assert len(result.state.skeptic_history) == 2
    assert result.state.skeptic_history[0].requires_revision is True
    assert result.state.skeptic_history[1].requires_revision is False

    # Verify Scientific Consensus & Contradiction Synthesizer
    assert len(result.state.conflicts) >= 1
    first_conflict = result.state.conflicts[0]
    assert first_conflict.support_authority_weight > 0
    assert first_conflict.counter_authority_weight > 0

    # Verify Tiered Model Routing & Cost/Latency Telemetry
    telemetry = result.state.cost_telemetry
    assert telemetry
    assert "fast" in telemetry["tier_usage"]
    assert "reasoning" in telemetry["tier_usage"]
    assert telemetry["tier_usage"]["fast"]["calls"] >= 2
    assert telemetry["tier_usage"]["reasoning"]["calls"] >= 4
    assert "verify" in telemetry["node_latencies"]
    assert "p95_ms" in telemetry["node_latencies"]["verify"]

    metrics = compute_metrics(result.state)
    assert metrics.total_claims >= 4
    assert metrics.supported == metrics.total_claims
    assert metrics.refuted == 0
    assert metrics.citation_accuracy == 1.0
    assert metrics.conflicts_resolved >= 1
    assert result.state.report_markdown is not None
    assert "## Adversarial Skeptic Audit Log" in result.state.report_markdown
    assert "## Scientific Disagreement & Consensus Analysis" in result.state.report_markdown

    # Verify every claim's EvidenceSpan slices the exact quote from the paper abstract
    for claim in result.state.claims:
        assert claim.verdict is Verdict.SUPPORTED
        assert claim.evidence
        for span in claim.evidence:
            paper = result.state.papers[span.paper_id]
            assert paper.abstract[span.char_start : span.char_end] == span.quote

    # Verify SQLite checkpointing and deterministic replay from node 'verify'
    ckpt = store.load_snapshot("test-demo-run", "extract_claims")
    assert ckpt is not None

    replayed = await replay_research(
        "test-demo-run",
        from_node="verify",
        settings=settings,
        llm=DemoLLM(inject_initial_overclaim=False),
        embedder=HashingEmbedder(dim=256),
        sources=(
            DemoSource(DEMO_PAPERS[:5], name="arxiv"),
            DemoSource(DEMO_PAPERS[3:], name="openalex"),
        ),
        trace_store=store,
        new_run_id="test-replay-run",
    )
    assert replayed.status is RunStatus.COMPLETED
    assert replayed.steps[0] == "verify"
    assert "contradiction" in replayed.steps
    assert replayed.steps[-1] == "render"

    runs = store.list_runs()
    assert len(runs) == 2
    events = store.get_events("test-demo-run")
    assert len(events) > 15
    store.close()
