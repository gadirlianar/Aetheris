"""High-level research pipeline assembly, tiered routing, and deterministic state replay."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import httpx

from aetheris.agents.claims import ClaimExtractor
from aetheris.agents.context import ResearchContext
from aetheris.agents.contradiction import ContradictionSynthesizer
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
from aetheris.llm.client import ModelAliasClient, TieredLLMRouter, TokenBucketRateLimiter
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

PIPELINE_NODE_ORDER: tuple[str, ...] = (
    "plan",
    "retrieve",
    "synthesize",
    "extract_claims",
    "verify",
    "skeptic",
    "contradiction",
    "render",
)


def route_after_skeptic(state: ResearchState) -> str:
    """Loop back to ``retrieve`` on Skeptic revision, else proceed to ``contradiction``."""
    if state.skeptic_history and state.skeptic_history[-1].requires_revision:
        return "retrieve"
    return "contradiction"


def build_research_graph() -> Graph[ResearchState, ResearchContext]:
    return (
        Graph[ResearchState, ResearchContext]()
        .add_node("plan", Planner(), retry=DEFAULT_RETRY)
        .add_node("retrieve", Retriever(), retry=DEFAULT_RETRY)
        .add_node("synthesize", Synthesizer(), retry=DEFAULT_RETRY)
        .add_node("extract_claims", ClaimExtractor(), retry=DEFAULT_RETRY)
        .add_node("verify", Verifier(), retry=DEFAULT_RETRY)
        .add_node("skeptic", Skeptic(), retry=DEFAULT_RETRY)
        .add_node("contradiction", ContradictionSynthesizer(), retry=DEFAULT_RETRY)
        .add_node("render", ReportRenderer())
        .add_edge("plan", "retrieve")
        .add_edge("retrieve", "synthesize")
        .add_edge("synthesize", "extract_claims")
        .add_edge("extract_claims", "verify")
        .add_edge("verify", "skeptic")
        .add_conditional("skeptic", route_after_skeptic, ("retrieve", "contradiction"))
        .add_edge("contradiction", "render")
        .add_edge("render", END)
    )


def _build_tiered_llm(
    cfg: Settings, llm: LLMClient | TieredLLMRouter | None
) -> LLMClient | TieredLLMRouter:
    if isinstance(llm, TieredLLMRouter):
        return llm
    if llm is not None:
        if llm.model == "aetheris-demo-1.0":
            return TieredLLMRouter(
                fast_client=ModelAliasClient(llm, "aetheris-demo-fast"),
                reasoning_client=ModelAliasClient(llm, "aetheris-demo-reasoning"),
            )
        return llm
    if cfg.gemini_api_key is None:
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Export GEMINI_API_KEY, use --demo, "
            "or pass `llm` and `embedder`."
        )
    key = cfg.gemini_api_key.get_secret_value()
    reasoning_model = cfg.llm_model if cfg.llm_model != "gemini-3.8-flash" else cfg.reasoning_model
    return TieredLLMRouter(
        fast_client=GeminiLLM(key, cfg.fast_model, thinking_level="low"),
        reasoning_client=GeminiLLM(key, reasoning_model, thinking_level=cfg.thinking_level),
    )


def _build_embedder(cfg: Settings, embedder: Embedder | None) -> Embedder:
    if embedder is not None:
        return embedder
    if cfg.gemini_api_key is None:
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Export GEMINI_API_KEY, use --demo, "
            "or pass `llm` and `embedder`."
        )
    key = cfg.gemini_api_key.get_secret_value()
    return GeminiEmbedder(key, cfg.embedding_model, dim=cfg.embedding_dim)


async def run_research(
    question: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | TieredLLMRouter | None = None,
    embedder: Embedder | None = None,
    sources: Sequence[Source] | None = None,
    trace_store: TraceStore | None = None,
    on_event: EventHandler | None = None,
    run_id: str | None = None,
) -> RunResult[ResearchState]:
    cfg = settings or Settings()
    rid = run_id or uuid.uuid4().hex[:12]

    active_llm = _build_tiered_llm(cfg, llm)
    active_emb = _build_embedder(cfg, embedder)

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
        max_cost_usd=cfg.max_cost_usd,
    )
    limiter = TokenBucketRateLimiter(cfg.rate_limit_rps, cfg.rate_limit_burst)
    inst_llm = InstrumentedLLM(active_llm, events, budget, rate_limiter=limiter)
    inst_emb = InstrumentedEmbedder(active_emb, events)
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


async def replay_research(
    run_id: str,
    *,
    from_node: str,
    trace_store: TraceStore,
    settings: Settings | None = None,
    llm: LLMClient | TieredLLMRouter | None = None,
    embedder: Embedder | None = None,
    sources: Sequence[Source] | None = None,
    on_event: EventHandler | None = None,
    new_run_id: str | None = None,
) -> RunResult[ResearchState]:
    """Deterministically resume or fork a persisted research run starting at ``from_node``."""
    cfg = settings or Settings()
    graph = build_research_graph()
    if from_node not in graph.nodes:
        valid = ", ".join(PIPELINE_NODE_ORDER)
        raise ValueError(f"Unknown node '{from_node}'. Valid nodes: {valid}")

    raw_snapshot = trace_store.load_state_before_node(run_id, from_node, PIPELINE_NODE_ORDER)
    if raw_snapshot is None:
        raise ValueError(f"Run '{run_id}' not found in trace store.")

    rid = new_run_id or f"{run_id}-replay-{from_node}"
    state = ResearchState.model_validate(raw_snapshot).model_copy(update={"run_id": rid})

    active_llm = _build_tiered_llm(cfg, llm)
    active_emb = _build_embedder(cfg, embedder)

    events = EventBus(rid)
    trace_store.start_run(rid, state.question)
    events.subscribe(trace_store.record)
    if on_event is not None:
        events.subscribe(on_event)

    budget = Budget(
        max_llm_calls=cfg.max_llm_calls,
        max_total_tokens=cfg.max_total_tokens,
        max_seconds=cfg.max_seconds,
        max_cost_usd=cfg.max_cost_usd,
    )
    limiter = TokenBucketRateLimiter(cfg.rate_limit_rps, cfg.rate_limit_burst)
    inst_llm = InstrumentedLLM(active_llm, events, budget, rate_limiter=limiter)
    inst_emb = InstrumentedEmbedder(active_emb, events)
    evidence = EvidenceStore(inst_emb, mode=cfg.retrieval_mode)

    # Rehydrate the EvidenceStore from the checkpointed papers
    if state.papers:
        chunks = await evidence.add_papers(list(state.papers.values()))
        state = state.model_copy(
            update={
                "chunks": {c.id: c for c in chunks},
                "knowledge_graph": evidence.graph_snapshot(),
            }
        )

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
        result = await Engine(events, budget=budget).run(graph, state, ctx, start_at=from_node)

    trace_store.finish_run(rid, result.state.model_dump(mode="json"))
    return result
