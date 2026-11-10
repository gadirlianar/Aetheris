"""Unit tests for text segmentation, BM25, RRF, GraphRAG PPR, and EvidenceStore."""

from __future__ import annotations

from aetheris.demo import DEMO_PAPERS
from aetheris.llm.hashing import HashingEmbedder
from aetheris.retrieval.bm25 import BM25Index
from aetheris.retrieval.fusion import reciprocal_rank_fusion
from aetheris.retrieval.graph import KnowledgeGraphIndex, extract_entities
from aetheris.retrieval.store import EvidenceStore
from aetheris.retrieval.text import chunk_paper, split_sentences, tokenize


def test_sentence_splitting_and_chunk_offsets() -> None:
    text = (
        "Prior work (e.g. Lewis et al. 2020) showed strong gains in Fig. 2. "
        "However, closed-book models still hallucinate on rare entities. "
        "GraphRAG bridges multi-hop facts across documents."
    )
    spans = split_sentences(text)
    assert len(spans) == 3
    assert text[spans[0][0] : spans[0][1]].startswith("Prior work")

    for paper in DEMO_PAPERS:
        chunks = chunk_paper(paper)
        assert chunks
        for ch in chunks:
            assert paper.abstract[ch.char_start : ch.char_end] == ch.text


def test_bm25_and_rrf() -> None:
    docs = [
        tokenize("retrieval augmented generation reduces hallucinations"),
        tokenize("alphaFold predicts 3D protein backbone coordinates"),
    ]
    bm25 = BM25Index(docs)
    scores = bm25.scores(tokenize("retrieval hallucinations"))
    assert scores[0] > scores[1]

    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    top_two = {fused[0][0], fused[1][0]}
    assert top_two == {"a", "b"}


async def test_knowledge_graph_and_evidence_store_modes() -> None:
    entities = extract_entities(
        "GraphRAG and Reciprocal Rank Fusion improve multi-hop retrieval on SciFact."
    )
    assert any("graphrag" in e for e in entities)

    kg = KnowledgeGraphIndex()
    papers = {p.id: p for p in DEMO_PAPERS[:6]}
    for p in DEMO_PAPERS[:6]:
        kg.add_chunks(chunk_paper(p), papers)

    ranked = kg.rank_chunks("GraphRAG reciprocal rank fusion multi-hop recall")
    assert ranked
    assert any("2404.16130" in cid for cid in ranked[:3])

    snap = kg.snapshot()
    assert snap.nodes
    assert snap.edges

    store = EvidenceStore(HashingEmbedder(dim=256), mode="graph_hybrid")
    await store.add_papers(DEMO_PAPERS)
    assert len(store) > 0

    for mode in ("dense", "bm25", "hybrid", "graph_hybrid"):
        hits = await store.search(
            "Prime editing Cas9 reverse transcriptase double-strand breaks",
            k=3,
            mode=mode,
        )
        assert hits
        assert hits[0].chunk.paper_id == "s2:3f4e5d6c7b8a"
