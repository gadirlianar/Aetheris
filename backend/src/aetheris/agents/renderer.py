"""Report renderer and verification metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aetheris.agents.context import ResearchContext
from aetheris.core.models import Paper, ResearchState, Verdict

_BADGE = {
    Verdict.SUPPORTED: "SUPPORTED",
    Verdict.REFUTED: "REFUTED",
    Verdict.NOT_ENOUGH_INFO: "NOT ENOUGH INFO",
}

_CONSENSUS_LABEL = {
    "strong_consensus_support": "STRONG CONSENSUS SUPPORT",
    "qualified_consensus": "QUALIFIED CONSENSUS (BOUNDARY-DEPENDENT)",
    "contested_methodology": "CONTESTED METHODOLOGY",
    "counter_evidence_prevails": "COUNTER-EVIDENCE PREVAILS",
}


@dataclass(frozen=True)
class VerificationMetrics:
    total_claims: int
    supported: int
    refuted: int
    not_enough_info: int
    support_rate: float
    citation_accuracy: float
    grounding_failures: int
    invalid_citations: int
    skeptic_rounds: int = 0
    conflicts_resolved: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(state: ResearchState) -> VerificationMetrics:
    claims = state.claims
    n = len(claims)
    sup = sum(c.verdict is Verdict.SUPPORTED for c in claims)
    ref = sum(c.verdict is Verdict.REFUTED for c in claims)
    nei = sum(c.verdict is Verdict.NOT_ENOUGH_INFO for c in claims)
    decided = [c for c in claims if c.decisive_evidence_cited is not None]
    cit_acc = (
        sum(bool(c.decisive_evidence_cited) for c in decided) / len(decided) if decided else 0.0
    )
    inv = sum(len(s.invalid_citations) for s in state.draft.iter_sentences()) if state.draft else 0
    raw_cost = state.cost_telemetry.get("total_cost_usd", 0.0)
    cost_usd = float(raw_cost) if isinstance(raw_cost, int | float) else 0.0
    return VerificationMetrics(
        total_claims=n,
        supported=sup,
        refuted=ref,
        not_enough_info=nei,
        support_rate=(sup / n) if n else 0.0,
        citation_accuracy=cit_acc,
        grounding_failures=sum(c.grounding_failed for c in claims),
        invalid_citations=inv,
        skeptic_rounds=state.skeptic_rounds,
        conflicts_resolved=len(state.conflicts),
        graph_nodes=len(state.knowledge_graph.nodes),
        graph_edges=len(state.knowledge_graph.edges),
        total_cost_usd=cost_usd,
    )


def render_markdown(state: ResearchState) -> str:
    if state.draft is None:
        raise ValueError("cannot render without a draft")

    bib: dict[str, int] = {}

    def ref(paper_id: str) -> int:
        if paper_id not in bib:
            bib[paper_id] = len(bib) + 1
        return bib[paper_id]

    for s in state.draft.iter_sentences():
        for cid in s.citations:
            if cid in state.chunks:
                ref(state.chunks[cid].paper_id)
    for c in state.claims:
        for span in c.evidence:
            ref(span.paper_id)
    for cf in state.conflicts:
        for src in [*cf.supporting_sources, *cf.counter_sources]:
            if src.paper_id in state.papers:
                ref(src.paper_id)

    m = compute_metrics(state)
    lines = [
        f"# {state.draft.title}",
        "",
        f"> **Research question:** {state.question}",
        ">",
        f"> **Verification summary:** {m.supported}/{m.total_claims} claims supported "
        f"({m.support_rate:.0%}), {m.refuted} refuted, {m.not_enough_info} insufficient evidence · "
        f"citation accuracy {m.citation_accuracy:.0%} · {len(bib)} sources cited "
        f"({len(state.papers)} retrieved) · {m.skeptic_rounds} skeptic audit round(s) · "
        f"{m.conflicts_resolved} disagreement resolution(s)",
        "",
    ]

    for section in state.draft.sections:
        lines += [f"## {section.heading}", ""]
        for s in section.sentences:
            refs = sorted(
                {ref(state.chunks[cid].paper_id) for cid in s.citations if cid in state.chunks}
            )
            tag = " " + "".join(f"[{n}]" for n in refs) if refs else ""
            lines.append(f"- **{s.id}.** {s.text}{tag}")
            for claim in state.claims_for(s.id):
                verdict = claim.verdict or Verdict.NOT_ENOUGH_INFO
                conf = f" ({claim.confidence:.2f})" if claim.confidence is not None else ""
                lines.append(f"  - `{_BADGE[verdict]}{conf}` _{claim.text}_")
                for span in claim.evidence:
                    p = state.papers.get(span.paper_id)
                    who = p.short_citation() if p else span.paper_id
                    lines.append(
                        f"    - [{ref(span.paper_id)}] ({who}, chars "
                        f'{span.char_start}–{span.char_end}, {span.match}): "{span.quote}"'
                    )
                if verdict is not Verdict.SUPPORTED and claim.rationale:
                    lines.append(f"    - Verifier note: {claim.rationale}")
                if claim.skeptic_notes:
                    lines.append(f"    - Skeptic critique: {claim.skeptic_notes}")
        lines.append("")

    if state.conflicts:
        lines += ["## Scientific Disagreement & Consensus Analysis", ""]
        for cf in state.conflicts:
            badge = _CONSENSUS_LABEL.get(cf.consensus_verdict, cf.consensus_verdict.upper())
            lines.append(
                f"- **[{cf.id}] {cf.topic}** — `{badge}` "
                f"(Support Authority $W={cf.support_authority_weight:.2f}$ vs. "
                f"Counter Authority $W={cf.counter_authority_weight:.2f}$, "
                f"Citation Ratio {cf.citation_ratio:.2f}x, $\\Delta t={cf.recency_gap_years}$y)"
            )
            for src in cf.supporting_sources:
                r_idx = ref(src.paper_id) if src.paper_id in state.papers else "?"
                lines.append(
                    f"  - *Primary / Supporting* [{r_idx}] {src.citation_label} "
                    f"({src.venue or 'preprint'}, {src.citation_count} citations, "
                    f'authority={src.authority_score:.2f}): "{src.quote}"'
                )
            for src in cf.counter_sources:
                r_idx = ref(src.paper_id) if src.paper_id in state.papers else "?"
                lines.append(
                    f"  - *Qualifying / Counter* [{r_idx}] {src.citation_label} "
                    f"({src.venue or 'preprint'}, {src.citation_count} citations, "
                    f'authority={src.authority_score:.2f}): "{src.quote}"'
                )
            lines.append(f"  - **Consensus Resolution:** {cf.synthesis}")
        lines.append("")

    if state.skeptic_history:
        lines += ["## Adversarial Skeptic Audit Log", ""]
        for entry in state.skeptic_history:
            status = "Revision Triggered" if entry.requires_revision else "Finalized"
            lines.append(f"- **Round {entry.round} ({status}):** {entry.summary}")
        lines.append("")

    if bib:
        lines += ["## References", ""]
        for paper_id, num in sorted(bib.items(), key=lambda kv: kv[1]):
            paper = state.papers.get(paper_id)
            if paper is not None:
                lines.append(_format_reference(num, paper))
    return "\n".join(lines).rstrip() + "\n"


def _format_reference(num: int, paper: Paper) -> str:
    authors = ", ".join(paper.authors[:4]) + (" et al." if len(paper.authors) > 4 else "")
    year = f" ({paper.year})" if paper.year else ""
    venue = f" *{paper.venue}*." if paper.venue else ""
    return f"{num}. {authors}{year}. **{paper.title}**.{venue} [{paper.id}]({paper.url})"


class ReportRenderer:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        telemetry = ctx.llm.budget.telemetry_summary()
        updated = state.model_copy(update={"cost_telemetry": telemetry})
        return updated.model_copy(update={"report_markdown": render_markdown(updated)})
