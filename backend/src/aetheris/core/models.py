"""Core domain models.

These are the *internal* representations shared by every agent. LLM output
schemas are intentionally kept separate (see each agent module): they are lean,
flat and free of numeric/array bounds so that constrained decoding stays cheap,
and they are mapped onto these richer models after validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceName = Literal["arxiv", "openalex", "semantic_scholar"]
RetrievalMode = Literal["dense", "bm25", "hybrid", "graph_hybrid"]
NodeKind = Literal["paper", "chunk", "entity"]
ConsensusVerdict = Literal[
    "strong_consensus_support",
    "qualified_consensus",
    "contested_methodology",
    "counter_evidence_prevails",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- sources


class Paper(_Frozen):
    """A scientific source document (title + abstract + bibliographic metadata)."""

    id: str = Field(description="Globally unique id, e.g. 'arxiv:1706.03762' or 'openalex:W123'.")
    source: SourceName
    title: str
    abstract: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str
    citation_count: int | None = None

    def short_citation(self) -> str:
        """Human-readable ``Surname et al., 2017`` style label."""
        if not self.authors:
            who = "Unknown"
        else:
            surname = self.authors[0].split()[-1]
            who = surname if len(self.authors) == 1 else f"{surname} et al."
        return f"{who}, {self.year}" if self.year else who


class Chunk(_Frozen):
    """A contiguous slice of a paper abstract. Offsets index into ``Paper.abstract``."""

    id: str
    paper_id: str
    text: str
    char_start: int
    char_end: int


# --------------------------------------------------------------------------- knowledge graph


class GraphNode(_Frozen):
    id: str
    label: str
    kind: NodeKind
    weight: float = 1.0
    paper_id: str | None = None


class GraphEdge(_Frozen):
    source: str
    target: str
    relation: str
    weight: float = 1.0


class KnowledgeGraphSnapshot(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# --------------------------------------------------------------------------- planning


class SubQuestion(BaseModel):
    question: str
    rationale: str
    search_queries: list[str]


class ResearchPlan(BaseModel):
    objective: str
    sub_questions: list[SubQuestion]

    def all_queries(self) -> list[str]:
        seen: dict[str, None] = {}
        for sq in self.sub_questions:
            for q in sq.search_queries:
                seen.setdefault(q.strip(), None)
        return [q for q in seen if q]


# --------------------------------------------------------------------------- report


class ReportSentence(BaseModel):
    id: str
    text: str
    citations: list[str] = Field(default_factory=list, description="Chunk ids cited.")
    invalid_citations: list[str] = Field(
        default_factory=list,
        description="Citation labels produced by the model that do not map to any evidence.",
    )


class ReportSection(BaseModel):
    heading: str
    sentences: list[ReportSentence]


class DraftReport(BaseModel):
    title: str
    sections: list[ReportSection]

    def iter_sentences(self) -> Iterator[ReportSentence]:
        for section in self.sections:
            yield from section.sentences

    def sentence(self, sentence_id: str) -> ReportSentence | None:
        return next((s for s in self.iter_sentences() if s.id == sentence_id), None)


# --------------------------------------------------------------------------- verification


class Verdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_INFO = "not_enough_info"


MatchKind = Literal["exact", "normalized", "fuzzy"]


class EvidenceSpan(_Frozen):
    """A quote that was mechanically located inside a chunk."""

    chunk_id: str
    paper_id: str
    quote: str
    char_start: int = Field(description="Offset into the paper abstract.")
    char_end: int
    match: MatchKind


class Claim(BaseModel):
    """An atomic, self-contained, checkable statement extracted from the report."""

    id: str
    sentence_id: str
    text: str
    verdict: Verdict | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    considered_chunk_ids: list[str] = Field(default_factory=list)
    grounding_failed: bool = Field(
        default=False,
        description="The verifier issued a verdict but none of its quotes exist in the evidence; "
        "the verdict was downgraded to NOT_ENOUGH_INFO.",
    )
    decisive_evidence_cited: bool | None = Field(
        default=None,
        description="For grounded verdicts: does the decisive evidence come from a chunk the "
        "synthesizer actually cited for this sentence? Basis of citation accuracy.",
    )
    skeptic_notes: str | None = Field(
        default=None,
        description="Critique or counter-evidence warning raised by the Skeptic agent.",
    )


class SkepticIssue(BaseModel):
    claim_id: str
    issue: str
    counter_query: str


class SkepticCritique(BaseModel):
    round: int
    summary: str
    requires_revision: bool
    issues: list[SkepticIssue] = Field(default_factory=list)
    counter_queries: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ contradiction & consensus


class WeightedEvidenceSource(_Frozen):
    """A source paper weighted by citation count, recency, and venue authority."""

    paper_id: str
    citation_label: str
    stance: Literal["supports", "refutes", "qualifies"]
    year: int | None = None
    citation_count: int = 0
    venue: str | None = None
    authority_score: float = Field(ge=0.0, le=1.0)
    quote: str


class ConflictResolution(BaseModel):
    """Structured analysis of competing or boundary-qualified findings across retrieved papers."""

    id: str
    topic: str
    claim_ids: list[str] = Field(default_factory=list)
    supporting_sources: list[WeightedEvidenceSource] = Field(default_factory=list)
    counter_sources: list[WeightedEvidenceSource] = Field(default_factory=list)
    support_authority_weight: float = 0.0
    counter_authority_weight: float = 0.0
    recency_gap_years: int = 0
    citation_ratio: float = 1.0
    consensus_verdict: ConsensusVerdict = "qualified_consensus"
    synthesis: str


# --------------------------------------------------------------------------- run state


class ResearchState(BaseModel):
    """The single, serialisable state object threaded through the graph."""

    run_id: str
    question: str
    plan: ResearchPlan | None = None
    papers: dict[str, Paper] = Field(default_factory=dict)
    chunks: dict[str, Chunk] = Field(default_factory=dict)
    knowledge_graph: KnowledgeGraphSnapshot = Field(default_factory=KnowledgeGraphSnapshot)
    executed_queries: list[str] = Field(default_factory=list)
    pending_queries: list[str] = Field(default_factory=list)
    draft: DraftReport | None = None
    claims: list[Claim] = Field(default_factory=list)
    skeptic_rounds: int = 0
    skeptic_history: list[SkepticCritique] = Field(default_factory=list)
    conflicts: list[ConflictResolution] = Field(default_factory=list)
    cost_telemetry: dict[str, Any] = Field(default_factory=dict)
    report_markdown: str | None = None

    def claims_for(self, sentence_id: str) -> list[Claim]:
        return [c for c in self.claims if c.sentence_id == sentence_id]
