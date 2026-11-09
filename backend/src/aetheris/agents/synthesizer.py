"""Synthesizer: retrieve evidence per sub-question and draft or revise a cited report."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aetheris.agents import prompts
from aetheris.agents.context import ResearchContext
from aetheris.agents.evidence import format_evidence
from aetheris.core.events import EventType
from aetheris.core.models import (
    Chunk,
    DraftReport,
    ReportSection,
    ReportSentence,
    ResearchState,
)


class _SentenceOut(BaseModel):
    text: str
    citations: list[str] = Field(
        description="Evidence labels such as 'E3' supporting the sentence."
    )


class _SectionOut(BaseModel):
    heading: str
    sentences: list[_SentenceOut]


class DraftOut(BaseModel):
    title: str
    sections: list[_SectionOut]


class Synthesizer:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        if state.plan is None:
            raise ValueError("synthesizer requires a plan")
        sub_questions = [sq.question for sq in state.plan.sub_questions]
        extra_queries = state.skeptic_history[-1].counter_queries if state.skeptic_history else []
        all_search_prompts = [*sub_questions, *extra_queries]
        hits = await ctx.evidence.search_many(
            all_search_prompts,
            k=ctx.settings.evidence_per_sub_question,
            mode=ctx.settings.retrieval_mode,
        )

        selected: dict[str, Chunk] = {}
        for per_query in hits:
            for hit in per_query:
                selected.setdefault(hit.chunk.id, hit.chunk)
        block = format_evidence(list(selected.values()), state.papers)

        skeptic_feedback = ""
        if state.skeptic_history:
            latest = state.skeptic_history[-1]
            issue_lines = [f"- Claim {iss.claim_id}: {iss.issue}" for iss in latest.issues]
            skeptic_feedback = (
                "\nSkeptic Audit (revise draft to fix these issues):\n"
                f"Summary: {latest.summary}\n"
                + ("\n".join(issue_lines) + "\n" if issue_lines else "")
            )

        result = await ctx.llm.generate_structured(
            system=prompts.SYNTHESIZER_SYSTEM,
            prompt=prompts.SYNTHESIZER_USER.format(
                question=state.question,
                sub_questions="\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1)),
                skeptic_feedback=skeptic_feedback,
                evidence=block.text,
            ),
            schema=DraftOut,
            purpose="synthesize",
        )

        sections: list[ReportSection] = []
        counter = 0
        invalid = 0
        for section in result.value.sections:
            sentences: list[ReportSentence] = []
            for sentence in section.sentences:
                if not sentence.text.strip():
                    continue
                counter += 1
                valid: list[str] = []
                bad: list[str] = []
                for label in sentence.citations:
                    chunk_id = block.resolve(label)
                    if chunk_id is None:
                        bad.append(label)
                    elif chunk_id not in valid:
                        valid.append(chunk_id)
                invalid += len(bad)
                sentences.append(
                    ReportSentence(
                        id=f"s{counter}",
                        text=sentence.text.strip(),
                        citations=valid,
                        invalid_citations=bad,
                    )
                )
            if sentences:
                sections.append(ReportSection(heading=section.heading.strip(), sentences=sentences))

        draft = DraftReport(title=result.value.title.strip(), sections=sections)
        await ctx.events.emit(
            EventType.PROGRESS,
            message=f"drafted {counter} sentences from {len(selected)} evidence chunks",
            sentences=counter,
            evidence_chunks=len(selected),
            invalid_citations=invalid,
        )
        return state.model_copy(update={"draft": draft})
