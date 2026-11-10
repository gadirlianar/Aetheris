"""Claim extractor: report sentences -> atomic, self-contained claims."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from aetheris.agents import prompts
from aetheris.agents.context import ResearchContext
from aetheris.core.events import EventType
from aetheris.core.models import Claim, ReportSection, ResearchState

MAX_CLAIMS_PER_SENTENCE = 4


class _ClaimOut(BaseModel):
    sentence_id: str
    claim: str


class ClaimsOut(BaseModel):
    claims: list[_ClaimOut]


class ClaimExtractor:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        if state.draft is None:
            raise ValueError("claim extractor requires a draft")
        per_section = await asyncio.gather(
            *(self._extract(section, state.question, ctx) for section in state.draft.sections)
        )
        claims: list[Claim] = []
        for raw in per_section:
            for sentence_id, text in raw:
                claims.append(Claim(id=f"c{len(claims) + 1}", sentence_id=sentence_id, text=text))
        await ctx.events.emit(
            EventType.PROGRESS, message=f"extracted {len(claims)} atomic claims", claims=len(claims)
        )
        return state.model_copy(update={"claims": claims})

    async def _extract(
        self, section: ReportSection, question: str, ctx: ResearchContext
    ) -> list[tuple[str, str]]:
        # Only sentences with citations carry factual content by construction of the
        # synthesizer prompt; uncited sentences are still sent so that unsupported
        # factual statements are caught rather than silently skipped.
        valid_ids = {s.id for s in section.sentences}
        result = await ctx.llm.generate_structured(
            system=prompts.CLAIM_EXTRACTOR_SYSTEM.format(
                max_claims_per_sentence=MAX_CLAIMS_PER_SENTENCE
            ),
            prompt=prompts.CLAIM_EXTRACTOR_USER.format(
                question=question,
                heading=section.heading,
                sentences="\n".join(f"[{s.id}] {s.text}" for s in section.sentences),
            ),
            schema=ClaimsOut,
            purpose="extract_claims",
            tier="fast",
        )
        per_sentence: dict[str, int] = {}
        out: list[tuple[str, str]] = []
        for item in result.value.claims:
            sid = item.sentence_id.strip().strip("[]")
            text = item.claim.strip()
            if sid not in valid_ids or not text:
                continue
            if per_sentence.get(sid, 0) >= MAX_CLAIMS_PER_SENTENCE:
                continue
            per_sentence[sid] = per_sentence.get(sid, 0) + 1
            out.append((sid, text))
        order = {s.id: i for i, s in enumerate(section.sentences)}
        out.sort(key=lambda pair: order[pair[0]])  # stable: keeps claim order within a sentence
        return out
