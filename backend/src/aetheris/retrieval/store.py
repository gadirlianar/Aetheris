"""Per-run 3-way hybrid + GraphRAG evidence index: Dense + BM25 + Knowledge Graph PPR."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from aetheris.core.models import Chunk, KnowledgeGraphSnapshot, Paper, RetrievalMode
from aetheris.llm.base import Embedder, Vector
from aetheris.retrieval.bm25 import BM25Index
from aetheris.retrieval.fusion import reciprocal_rank_fusion
from aetheris.retrieval.graph import KnowledgeGraphIndex
from aetheris.retrieval.text import chunk_paper, tokenize


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    graph_rank: int | None = None


class EvidenceStore:
    """Holds every chunk retrieved during a run and answers hybrid / GraphRAG queries.

    Supports four retrieval modes for systematic ablation studies:
      * ``"dense"``        — dense cosine similarity only
      * ``"bm25"``         — Okapi BM25+ lexical similarity only
      * ``"hybrid"``       — 2-way Reciprocal Rank Fusion (Dense + BM25)
      * ``"graph_hybrid"`` — 3-way Reciprocal Rank Fusion (Dense + BM25 + GraphRAG PPR)
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        candidate_pool: int = 50,
        mode: RetrievalMode = "graph_hybrid",
    ) -> None:
        self._embedder = embedder
        self._candidate_pool = candidate_pool
        self._mode: RetrievalMode = mode
        self._chunks: list[Chunk] = []
        self._ids: dict[str, int] = {}
        self._papers: dict[str, Paper] = {}
        self._titles: dict[str, str] = {}
        self._vectors: Vector = np.zeros((0, embedder.dim), dtype=np.float32)
        self._tokens: list[list[str]] = []
        self._bm25: BM25Index | None = None
        self._kg = KnowledgeGraphIndex()
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> RetrievalMode:
        return self._mode

    @mode.setter
    def mode(self, value: RetrievalMode) -> None:
        self._mode = value

    @property
    def knowledge_graph(self) -> KnowledgeGraphIndex:
        return self._kg

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._ids

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[self._ids[chunk_id]]

    def graph_snapshot(self, *, max_entities: int = 36) -> KnowledgeGraphSnapshot:
        return self._kg.snapshot(max_entities=max_entities)

    async def add_papers(self, papers: Iterable[Paper]) -> list[Chunk]:
        """Chunk, embed, and index papers into Dense, BM25, and KnowledgeGraph indices."""
        async with self._lock:
            new: list[Chunk] = []
            for paper in papers:
                self._papers[paper.id] = paper
                self._titles[paper.id] = paper.title
                new.extend(c for c in chunk_paper(paper) if c.id not in self._ids)
            if not new:
                return []
            vectors = await self._embedder.embed_documents(
                [(self._titles[c.paper_id], c.text) for c in new]
            )
            for chunk in new:
                self._ids[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
                self._tokens.append(tokenize(f"{self._titles[chunk.paper_id]} {chunk.text}"))
            self._vectors = np.vstack([self._vectors, vectors])
            self._bm25 = BM25Index(self._tokens)
            self._kg.add_chunks(new, self._papers)
            return new

    async def search(
        self,
        query: str,
        *,
        k: int = 8,
        max_per_paper: int = 2,
        exclude: Sequence[str] = (),
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_vec = (await self._embedder.embed_queries([query]))[0]
        return self.search_with_vector(
            query, query_vec, k=k, max_per_paper=max_per_paper, exclude=exclude, mode=mode
        )

    def search_with_vector(
        self,
        query: str,
        query_vec: Vector,
        *,
        k: int = 8,
        max_per_paper: int = 2,
        exclude: Sequence[str] = (),
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        assert self._bm25 is not None
        active_mode = mode or self._mode
        pool = min(self._candidate_pool, len(self._chunks))

        dense = self._vectors @ query_vec
        dense_order = np.argsort(-dense, kind="stable")[:pool]
        sparse = self._bm25.scores(tokenize(query))
        sparse_order = [i for i in np.argsort(-sparse, kind="stable")[:pool] if sparse[i] > 0]

        dense_ids = [self._chunks[i].id for i in dense_order]
        sparse_ids = [self._chunks[i].id for i in sparse_order]
        graph_ids = self._kg.rank_chunks(query, top_k=pool) if active_mode == "graph_hybrid" else []

        dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
        sparse_rank = {cid: r for r, cid in enumerate(sparse_ids, 1)}
        graph_rank = {cid: r for r, cid in enumerate(graph_ids, 1)}

        if active_mode == "dense":
            rankings: list[Sequence[str]] = [dense_ids]
        elif active_mode == "bm25":
            rankings = [sparse_ids or dense_ids]
        elif active_mode == "hybrid":
            rankings = [dense_ids, sparse_ids]
        else:
            rankings = [dense_ids, sparse_ids, graph_ids] if graph_ids else [dense_ids, sparse_ids]

        excluded = set(exclude)
        per_paper: dict[str, int] = {}
        results: list[ScoredChunk] = []
        for chunk_id, score in reciprocal_rank_fusion(rankings):
            if chunk_id in excluded:
                continue
            chunk = self.get(chunk_id)
            if per_paper.get(chunk.paper_id, 0) >= max_per_paper:
                continue
            per_paper[chunk.paper_id] = per_paper.get(chunk.paper_id, 0) + 1
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    graph_rank=graph_rank.get(chunk_id),
                )
            )
            if len(results) >= k:
                break
        return results

    async def search_many(
        self,
        queries: Sequence[str],
        *,
        k: int = 8,
        max_per_paper: int = 2,
        mode: RetrievalMode | None = None,
    ) -> list[list[ScoredChunk]]:
        """Batch variant: one embedding request for all queries."""
        if not self._chunks or not queries:
            return [[] for _ in queries]
        vectors = await self._embedder.embed_queries(queries)
        return [
            self.search_with_vector(q, v, k=k, max_per_paper=max_per_paper, mode=mode)
            for q, v in zip(queries, vectors, strict=True)
        ]
