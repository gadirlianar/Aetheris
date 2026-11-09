"""High-level research pipeline assembly."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import httpx

from aetheris.agents.claims import ClaimExtractor
from aetheris.agents.context import ResearchContext
from aetheris.agents.planner import Planner
from aetheris.agents.renderer import ReportRenderer
from aetheris.agents.retriever import Retriever
from aetheris.agents.skeptic import Skeptic
from aetheris.agents.synthesizer import Synthesizer
from aetheris.agents.verifier import Verifier
from aetheris.core.budget import Budget
from aetheris.core.config import Settings
from aetheris.core.errors import ConfigurationError
from aetheris.core.events import EventBus, EventHandler
from aetheris.core.models import ResearchState
from aetheris.llm.base import Embedder, LLMClient
from aetheris.llm.gemini import GeminiEmbedder, GeminiLLM
from aetheris.llm.instrumented import InstrumentedEmbedder, InstrumentedLLM
from aetheris.orchestrator.engine import Engine, RunResult
from aetheris.orchestrator.graph import END, Graph, RetryPolicy
from aetheris.retrieval.store import EvidenceStore
from aetheris.sources.arxiv import ArxivSource
from aetheris.sources.base import USER_AGENT, Source
from aetheris.sources.openalex import OpenAlexSource
from aetheris.sources.semantic_scholar import SemanticScholarSource
from aetheris.tracing.store import TraceStore

DEFAULT_RETRY = RetryPolicy(max_attempts=3, initial_delay=1.0)


def route_after_skeptic(state: ResearchState) -> str:
    """Conditional transition: loop back to ``retrieve`` on Skeptic revision, else ``render``."""
    if state.skeptic_history and state.skeptic_history[-1].requires_revision:
        return "retrieve"
    return "render"


def build_research_graph() -> Graph[ResearchState, ResearchContext]:
    return (
        Graph[ResearchState, ResearchContext]()
        .add_node("plan", Planner(), retry=DEFAULT_RETRY)
        .add_node("retrieve", Retriever(), retry=DEFAULT_RETRY)
        .add_node("synthesize", Synthesizer(), retry=DEFAULT_RETRY)
        .add_node("extract_claims", ClaimExtractor(), retry=DEFAULT_RETRY)
        .add_node("verify", Verifier(), retry=DEFAULT_RETRY)
        .add_node("skeptic", Skeptic(), retry=DEFAULT_RETRY)
        .add_node("render", ReportRenderer())
        .add_edge("plan", "retrieve")
        .add_edge("retrieve", "synthesize")
        .add_edge("synthesize", "extract_claims")
        .add_edge("extract_claims", "verify")
        .add_edge("verify", "skeptic")
        .add_conditional("skeptic", route_after_skeptic, ("retrieve", "render"))
        .add_edge("render", END)
    )


async def run_research(
    question: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    sources: Sequence[Source] | None = None,
    trace_store: TraceStore | None = None,
    on_event: EventHandler | None = None,
    run_id: str | None = None,
) -> RunResult[ResearchState]:
    cfg = settings or Settings()
    rid = run_id or uuid.uuid4().hex[:12]

    if llm is None or embedder is None:
        if cfg.gemini_api_key is None:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Export GEMINI_API_KEY, use --demo, "
                "or pass `llm` and `embedder`."
            )
        key = cfg.gemini_api_key.get_secret_value()
        llm = llm or GeminiLLM(key, cfg.llm_model, thinking_level=cfg.thinking_level)
        embedder = embedder or GeminiEmbedder(key, cfg.embedding_model, dim=cfg.embedding_dim)

    events = EventBus(rid)
    if trace_store is not None:
        trace_store.start_run(rid, question)
        events.subscribe(trace_store.record)
    if on_event is not None:
        events.subscribe(on_event)

    budget = Budget(
        max_llm_calls=cfg.max_llm_calls,
        max_total_tokens=cfg.max_total_tokens,
        max_seconds=cfg.max_seconds,
    )
    inst_llm = InstrumentedLLM(llm, events, budget)
    inst_emb = InstrumentedEmbedder(embedder, events)
    evidence = EvidenceStore(inst_emb, mode=cfg.retrieval_mode)

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=cfg.source_timeout_s,
        follow_redirects=True,
    ) as http:
        default_sources: list[Source] = [
            OpenAlexSource(
                http,
                email=cfg.openalex_email,
                api_key=cfg.openalex_api_key.get_secret_value() if cfg.openalex_api_key else None,
            ),
            ArxivSource(http),
        ]
        if cfg.enable_semantic_scholar:
            default_sources.append(
                SemanticScholarSource(
                    http,
                    api_key=(
                        cfg.semantic_scholar_api_key.get_secret_value()
                        if cfg.semantic_scholar_api_key
                        else None
                    ),
                )
            )
        active_sources: Sequence[Source] = sources or tuple(default_sources)
        ctx = ResearchContext(
            settings=cfg, llm=inst_llm, evidence=evidence, sources=active_sources, events=events
        )
        state = ResearchState(run_id=rid, question=question)
        result = await Engine(events, budget=budget).run(build_research_graph(), state, ctx)

    if trace_store is not None:
        trace_store.finish_run(rid, result.state.model_dump(mode="json"))
    return result
