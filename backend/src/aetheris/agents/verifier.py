"""Verifier: independently judge every claim against retrieved evidence in parallel batches.

Design principles:
  * isolation       – the verifier never sees the draft, only the claim and evidence,
                      so it cannot be anchored by the writer's framing;
  * bounded parallelism & rate limiting – claims are verified concurrently in batches
                      governed by ``asyncio.Semaphore(ctx.settings.max_concurrent_claims)``
                      and a ``TokenBucketRateLimiter``, preventing sequential latency
                      blowups on long reports while respecting upstream quota limits;
  * augmented evidence – besides the chunks the writer cited, fresh hybrid/GraphRAG
                      search results for the claim are included, which lets the verifier
                      find support the writer missed *and* detect mis-citations;
  * mechanical grounding – a verdict stands only if at least one of its quotes is
                      found in the evidence (see ``grounding.py``), unless disabled for
                      ablation benchmarking.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel

from aetheris.agents import prompts
from aetheris.agents.context import ResearchContext
from aetheris.agents.evidence import format_evidence
from aetheris.agents.grounding import locate_quote
from aetheris.core.events import EventType
from aetheris.core.models import Chunk, Claim, EvidenceSpan, ResearchState, Verdict
from aetheris.llm.client import TokenBucketRateLimiter


class _QuoteOut(BaseModel):
    evidence: str
    quote: str


class VerificationOut(BaseModel):
    verdict: Literal["supported", "refuted", "not_enough_info"]
    confidence: float
    rationale: str
    quotes: list[_QuoteOut]


class Verifier:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        if state.draft is None:
            raise ValueError("verifier requires a draft")
        if not state.claims:
            return state

        searches = await ctx.evidence.search_many(
            [c.text for c in state.claims],
            k=ctx.settings.evidence_per_claim,
            mode=ctx.settings.retrieval_mode,
        )
        max_concurrent = max(1, ctx.settings.max_concurrent_claims)
        semaphore = asyncio.Semaphore(max_concurrent)
        limiter = TokenBucketRateLimiter(
            rate_per_sec=ctx.settings.rate_limit_rps,
            capacity=ctx.settings.rate_limit_burst,
        )

        async def verify_with_limits(claim: Claim, hits: list[Chunk]) -> Claim:
            sentence = state.draft.sentence(claim.sentence_id) if state.draft else None
            cited = [
                ctx.evidence.get(cid)
                for cid in (sentence.citations if sentence else [])
                if cid in ctx.evidence
            ]
            candidates = list({c.id: c for c in [*cited, *hits]}.values())
            async with semaphore:
                await limiter.acquire(1.0)
                return await self._verify_one(claim, candidates, {c.id for c in cited}, state, ctx)

        # Process claims in concurrent batches while preserving deterministic claim order
        pairs = [
            (claim, [h.chunk for h in hits])
            for claim, hits in zip(state.claims, searches, strict=True)
        ]
        batch_size = max(max_concurrent * 2, 10)
        verified: list[Claim] = []
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            batch_results = await asyncio.gather(*(verify_with_limits(c, h) for c, h in batch))
            verified.extend(batch_results)

        counts = {v.value: sum(c.verdict is v for c in verified) for v in Verdict}
        msg = (
            f"verified {len(verified)} claims (concurrency={max_concurrent}): "
            f"{counts['supported']} supported, {counts['refuted']} refuted, "
            f"{counts['not_enough_info']} not enough info"
        )
        await ctx.events.emit(
            EventType.PROGRESS,
            message=msg,
            verdicts=counts,
            concurrency=max_concurrent,
            grounding_failures=sum(c.grounding_failed for c in verified),
        )
        return state.model_copy(update={"claims": verified})

    async def _verify_one(
        self,
        claim: Claim,
        candidates: list[Chunk],
        cited_ids: set[str],
        state: ResearchState,
        ctx: ResearchContext,
    ) -> Claim:
        considered = [c.id for c in candidates]
        if not candidates:
            return claim.model_copy(
                update={
                    "verdict": Verdict.NOT_ENOUGH_INFO,
                    "rationale": "No evidence available.",
                    "considered_chunk_ids": considered,
                }
            )
        block = format_evidence(candidates, state.papers)
        result = await ctx.llm.generate_structured(
            system=prompts.VERIFIER_SYSTEM,
            prompt=prompts.VERIFIER_USER.format(claim=claim.text, evidence=block.text),
            schema=VerificationOut,
            purpose="verify",
            tier="reasoning",
        )
        out = result.value
        by_id = {c.id: c for c in candidates}

        spans: list[EvidenceSpan] = []
        for q in out.quotes:
            chunk_id = block.resolve(q.evidence)
            if chunk_id is None:
                continue
            chunk = by_id[chunk_id]
            located = locate_quote(q.quote, chunk.text)
            if located is None:
                continue
            start, end, kind = located
            span = EvidenceSpan(
                chunk_id=chunk.id,
                paper_id=chunk.paper_id,
                quote=chunk.text[start:end],
                char_start=chunk.char_start + start,
                char_end=chunk.char_start + end,
                match=kind,
            )
            if span not in spans:
                spans.append(span)

        verdict = Verdict(out.verdict)
        confidence: float | None = min(1.0, max(0.0, out.confidence))
        grounding_failed = False
        if verdict is not Verdict.NOT_ENOUGH_INFO and not spans:
            grounding_failed = True
            if ctx.settings.enable_grounding_guard:
                verdict, confidence = Verdict.NOT_ENOUGH_INFO, None

        decisive_cited = (
            any(s.chunk_id in cited_ids for s in spans)
            if verdict is not Verdict.NOT_ENOUGH_INFO and spans
            else None
        )
        return claim.model_copy(
            update={
                "verdict": verdict,
                "confidence": confidence,
                "rationale": out.rationale.strip(),
                "evidence": spans,
                "considered_chunk_ids": considered,
                "grounding_failed": grounding_failed,
                "decisive_evidence_cited": decisive_cited,
            }
        )
