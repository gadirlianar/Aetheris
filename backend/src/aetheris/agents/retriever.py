"""Retriever: fan out planned or Skeptic-issued queries to sources, de-duplicate, and index."""

from __future__ import annotations

import asyncio
import time

from aetheris.agents.context import ResearchContext
from aetheris.core.errors import NoEvidenceError, TransientError
from aetheris.core.events import EventType
from aetheris.core.models import Paper, ResearchState
from aetheris.sources.base import Source
from aetheris.sources.dedup import merge_papers


class Retriever:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        if state.plan is None:
            raise ValueError("retriever requires a plan")
        candidate_queries = state.pending_queries or state.plan.all_queries()
        queries = [q for q in candidate_queries if q not in state.executed_queries]

        jobs = [(source, query) for query in queries for source in ctx.sources]
        outcomes = await asyncio.gather(*(self._query(src, q, ctx) for src, q in jobs))

        papers = dict(state.papers)
        new_papers = merge_papers(papers, (p for batch, _ in outcomes for p in batch))
        if not papers:
            if any(isinstance(err, TransientError) for _, err in outcomes):
                raise TransientError("every source query failed transiently")
            raise NoEvidenceError("no papers found for any planned query")

        new_chunks = await ctx.evidence.add_papers(new_papers)
        kg_snapshot = ctx.evidence.graph_snapshot()
        await ctx.events.emit(
            EventType.PROGRESS,
            message=(
                f"indexed {len(new_papers)} new papers ({len(new_chunks)} chunks, "
                f"{len(kg_snapshot.nodes)} graph nodes)"
            ),
            papers=len(papers),
            chunks=len(state.chunks) + len(new_chunks),
            graph_nodes=len(kg_snapshot.nodes),
            graph_edges=len(kg_snapshot.edges),
        )
        return state.model_copy(
            update={
                "papers": papers,
                "chunks": {**state.chunks, **{c.id: c for c in new_chunks}},
                "knowledge_graph": kg_snapshot,
                "executed_queries": [*state.executed_queries, *queries],
                "pending_queries": [],
            }
        )

    async def _query(
        self, source: Source, query: str, ctx: ResearchContext
    ) -> tuple[list[Paper], Exception | None]:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(ctx.settings.source_timeout_s * 3):
                papers = await source.search(query, limit=ctx.settings.papers_per_query)
        except Exception as exc:
            error: Exception = (
                exc if not isinstance(exc, TimeoutError) else TransientError("source timeout")
            )
            await ctx.events.emit(
                EventType.SOURCE_QUERY,
                source=source.name,
                query=query,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            return [], error
        await ctx.events.emit(
            EventType.SOURCE_QUERY,
            source=source.name,
            query=query,
            ok=True,
            results=len(papers),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return papers, None
