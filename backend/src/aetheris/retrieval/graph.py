"""GraphRAG: Entity-Relation Knowledge Graph & Personalized PageRank retrieval.

Builds a heterogeneous graph over:
  * ``paper:<id>``  — scientific source papers
  * ``chunk:<id>``  — abstract sentence-window chunks
  * ``entity:<id>`` — scientific concepts, methods, benchmarks, genes, and compounds

At query time, matching entities in the query seed a Personalized PageRank (PPR)
random walk (``nx.pagerank``). Probability mass diffuses across co-occurring entities
and multi-hop paper links into ``chunk:*`` nodes, surfacing evidence chunks that
bridge concepts across papers even when exact lexical overlap is sparse.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable

import networkx as nx

from aetheris.core.models import (
    Chunk,
    GraphEdge,
    GraphNode,
    KnowledgeGraphSnapshot,
    Paper,
)
from aetheris.retrieval.text import split_sentences, tokenize

_ACADEMIC_STOP = frozenset(
    {
        "paper",
        "study",
        "studies",
        "show",
        "shows",
        "shown",
        "demonstrate",
        "demonstrates",
        "propose",
        "proposed",
        "method",
        "methods",
        "approach",
        "approaches",
        "result",
        "results",
        "model",
        "models",
        "system",
        "systems",
        "work",
        "works",
        "based",
        "novel",
        "new",
        "state",
        "art",
        "existing",
        "previous",
        "however",
        "moreover",
        "furthermore",
        "therefore",
        "thus",
        "also",
        "First",
        "second",
        "finally",
        "evaluate",
        "evaluated",
        "evaluation",
        "experiment",
        "experiments",
        "experimental",
        "significant",
        "significantly",
        "improve",
        "improves",
        "improved",
        "performance",
        "achieve",
        "achieves",
        "achieved",
        "provide",
        "provides",
        "analysis",
        "problem",
        "problems",
        "task",
        "tasks",
        "application",
        "applications",
        "data",
        "dataset",
        "datasets",
        "large",
        "small",
        "high",
        "low",
        "different",
        "various",
        "several",
        "multiple",
        "general",
        "specific",
        "particular",
        "important",
        "effective",
        "efficient",
        "recent",
        "recently",
        "current",
        "standard",
        "common",
        "typical",
        "able",
        "et",
        "al",
        "fig",
        "table",
        "section",
        "compared",
        "compare",
        "comparison",
    }
)

_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]{1,15}(?:\s+[0-9]+(?:\.[0-9]+)?)?\b")
_HYPHEN_RE = re.compile(r"\b[a-zA-Z]{3,}(?:-[a-zA-Z0-9]{2,})+\b")


def _personalized_pagerank(
    graph: nx.Graph,
    personalization: dict[str, float],
    *,
    alpha: float = 0.82,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Exact power-iteration Personalized PageRank (no SciPy dependency required)."""
    nodes = list(graph.nodes())
    if not nodes or not personalization:
        return {}
    total_p = sum(max(0.0, v) for v in personalization.values())
    if total_p <= 0.0:
        return {}
    p = {n: max(0.0, personalization.get(n, 0.0)) / total_p for n in nodes}

    # Precompute weighted outgoing transitions per node
    transitions: dict[str, list[tuple[str, float]]] = {}
    dangling: list[str] = []
    for u in nodes:
        nbrs = graph[u]
        deg = sum(float(attrs.get("weight", 1.0)) for attrs in nbrs.values())
        if deg <= 0.0:
            dangling.append(u)
            transitions[u] = []
        else:
            transitions[u] = [
                (str(v), float(attrs.get("weight", 1.0)) / deg) for v, attrs in nbrs.items()
            ]

    rank = dict(p)
    teleport = 1.0 - alpha
    for _ in range(max_iter):
        dangling_mass = alpha * sum(rank[u] for u in dangling)
        nxt = {n: (teleport + dangling_mass) * p[n] for n in nodes}
        for u, edges in transitions.items():
            ru = alpha * rank[u]
            if ru == 0.0:
                continue
            for v, prob in edges:
                nxt[v] += ru * prob
        err = sum(abs(nxt[n] - rank[n]) for n in nodes)
        rank = nxt
        if err < tol:
            break
    return rank


def extract_entities(text: str, *, max_entities: int = 18) -> list[str]:
    """Extract canonical scientific concept identifiers from ``text``."""
    found: dict[str, float] = {}

    # 1. Technical acronyms, model names, biomedical symbols (e.g., RAG, BM25, BRCA1, GPT-4)
    for match in _ACRONYM_RE.findall(text):
        cleaned = match.strip()
        low = cleaned.lower()
        if low in _ACADEMIC_STOP or len(low) < 2:
            continue
        if any(c.isupper() for c in cleaned[1:]) or any(c.isdigit() for c in cleaned):
            found[low] = max(found.get(low, 0.0), 2.5)

    # 2. Hyphenated technical compounds (e.g., retrieval-augmented, closed-book, fine-tuning)
    for match in _HYPHEN_RE.findall(text):
        low = match.lower()
        parts = low.split("-")
        if all(p not in _ACADEMIC_STOP for p in parts):
            found[low] = max(found.get(low, 0.0), 2.2)

    # 3. Meaningful domain tokens & adjacent technical bigrams
    tokens = [t for t in tokenize(text) if t not in _ACADEMIC_STOP and len(t) >= 4]
    counts = Counter(tokens)
    for tok, freq in counts.items():
        score = 1.0 + 0.3 * min(freq, 3) + (0.4 if "-" in tok else 0.0)
        found[tok] = max(found.get(tok, 0.0), score)

    for a, b in itertools.pairwise(tokens):
        if a != b and len(a) >= 4 and len(b) >= 4:
            bigram = f"{a} {b}"
            found[bigram] = max(found.get(bigram, 0.0), 1.6)

    ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:max_entities]]


class KnowledgeGraphIndex:
    """In-memory heterogeneous knowledge graph with Personalized PageRank scoring."""

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._entity_df: Counter[str] = Counter()
        self._paper_titles: dict[str, str] = {}
        self._num_chunks = 0

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def add_chunks(self, chunks: Iterable[Chunk], papers: dict[str, Paper]) -> None:
        for pid, paper in papers.items():
            self._paper_titles[pid] = paper.title
            pnode = f"paper:{pid}"
            if not self._graph.has_node(pnode):
                self._graph.add_node(
                    pnode, label=paper.short_citation(), kind="paper", paper_id=pid
                )

        for chunk in chunks:
            cnode = f"chunk:{chunk.id}"
            if self._graph.has_node(cnode):
                continue
            self._num_chunks += 1
            pnode = f"paper:{chunk.paper_id}"
            if not self._graph.has_node(pnode):
                self._graph.add_node(
                    pnode, label=chunk.paper_id, kind="paper", paper_id=chunk.paper_id
                )
            self._graph.add_node(cnode, label=chunk.id, kind="chunk", paper_id=chunk.paper_id)
            self._graph.add_edge(pnode, cnode, relation="contains", weight=1.0)

            title = self._paper_titles.get(chunk.paper_id, "")
            entities = extract_entities(f"{title}. {chunk.text}")
            for ent in entities:
                enode = f"entity:{ent}"
                if not self._graph.has_node(enode):
                    self._graph.add_node(enode, label=ent, kind="entity", paper_id=None)
                self._entity_df[ent] += 1
                self._graph.add_edge(cnode, enode, relation="mentions", weight=1.5)

            # Sentence-level entity co-occurrence edges create multi-hop bridges
            for s_start, s_end in split_sentences(chunk.text):
                sent_entities = extract_entities(chunk.text[s_start:s_end], max_entities=8)
                for e1, e2 in itertools.combinations(sent_entities, 2):
                    n1, n2 = f"entity:{e1}", f"entity:{e2}"
                    if self._graph.has_node(n1) and self._graph.has_node(n2):
                        if self._graph.has_edge(n1, n2):
                            self._graph[n1][n2]["weight"] += 1.0
                        else:
                            self._graph.add_edge(n1, n2, relation="co_occurs", weight=1.0)

    def rank_chunks(self, query: str, *, top_k: int = 50) -> list[str]:
        """Rank chunk IDs by Personalized PageRank seeded from query entities."""
        if self._num_chunks == 0:
            return []

        query_entities = set(extract_entities(query, max_entities=12))
        query_tokens = set(tokenize(query))

        personalization: dict[str, float] = {}
        n_chunks = max(1, self._num_chunks)
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("kind") != "entity":
                continue
            label = str(attrs.get("label", ""))
            df = self._entity_df.get(label, 1)
            idf = math.log(1.0 + n_chunks / df)
            if label in query_entities:
                personalization[node] = 3.0 * idf
            elif any(tok in label.split() for tok in query_tokens):
                personalization[node] = 1.2 * idf

        if not personalization:
            return []

        scores = _personalized_pagerank(self._graph, personalization, alpha=0.82)
        chunk_scores = [
            (node.removeprefix("chunk:"), sc)
            for node, sc in scores.items()
            if node.startswith("chunk:") and sc > 0.0
        ]
        chunk_scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in chunk_scores[:top_k]]

    def snapshot(self, *, max_entities: int = 36) -> KnowledgeGraphSnapshot:
        """Export a concise, visualisable subgraph (papers + top entities + their edges)."""
        if self._graph.number_of_nodes() == 0:
            return KnowledgeGraphSnapshot()

        centrality: dict[str, float] = nx.degree_centrality(self._graph)
        paper_nodes = [n for n, d in self._graph.nodes(data=True) if d.get("kind") == "paper"]
        entity_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("kind") == "entity" and self._graph.degree(n) >= 2
        ]
        if not entity_nodes:
            entity_nodes = [n for n, d in self._graph.nodes(data=True) if d.get("kind") == "entity"]
        entity_nodes.sort(key=lambda n: (-centrality.get(n, 0.0), n))
        kept_entities = set(entity_nodes[:max_entities])

        nodes: list[GraphNode] = []
        for pnode in paper_nodes:
            attrs = self._graph.nodes[pnode]
            nodes.append(
                GraphNode(
                    id=pnode,
                    label=str(attrs.get("label", pnode)),
                    kind="paper",
                    weight=round(1.0 + centrality.get(pnode, 0.0) * 5.0, 3),
                    paper_id=attrs.get("paper_id"),
                )
            )
        for enode in entity_nodes[:max_entities]:
            attrs = self._graph.nodes[enode]
            nodes.append(
                GraphNode(
                    id=enode,
                    label=str(attrs.get("label", enode)),
                    kind="entity",
                    weight=round(1.0 + centrality.get(enode, 0.0) * 8.0, 3),
                )
            )

        edges: list[GraphEdge] = []
        seen_paper_entity: set[tuple[str, str]] = set()
        # Project paper -> chunk -> entity paths into direct paper -> entity edges for clean UI
        for pnode in paper_nodes:
            for cnode in self._graph.neighbors(pnode):
                if not str(cnode).startswith("chunk:"):
                    continue
                for enode in self._graph.neighbors(cnode):
                    if enode in kept_entities and (pnode, enode) not in seen_paper_entity:
                        seen_paper_entity.add((pnode, enode))
                        edges.append(
                            GraphEdge(source=pnode, target=enode, relation="discusses", weight=1.0)
                        )

        for u, v, data in self._graph.edges(data=True):
            if u in kept_entities and v in kept_entities:
                edges.append(
                    GraphEdge(
                        source=str(u),
                        target=str(v),
                        relation=str(data.get("relation", "co_occurs")),
                        weight=float(data.get("weight", 1.0)),
                    )
                )

        return KnowledgeGraphSnapshot(nodes=nodes, edges=edges)
