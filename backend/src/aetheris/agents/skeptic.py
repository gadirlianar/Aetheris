"""Adversarial Skeptic (Devil's Advocate) agent.

Audits verified claims for refutations, ungrounded assertions, or low confidence.
When challenged claims are found and the Skeptic budget allows another round, it
generates targeted literature queries and routes the state machine back to
``retrieve -> synthesize -> extract_claims -> verify``.
"""

from __future__ import annotations

from pydantic import BaseModel

from aetheris.agents import prompts
from aetheris.agents.context import ResearchContext
from aetheris.core.events import EventType
from aetheris.core.models import (
    Claim,
    ResearchState,
    SkepticCritique,
    SkepticIssue,
    Verdict,
)


class _IssueOut(BaseModel):
    claim_id: str
    issue: str
    counter_query: str


class SkepticOut(BaseModel):
    summary: str
    issues: list[_IssueOut]


class Skeptic:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        min_conf = ctx.settings.skeptic_min_confidence
        challenged = [
            c
            for c in state.claims
            if c.verdict in (Verdict.REFUTED, Verdict.NOT_ENOUGH_INFO)
            or (c.confidence is not None and c.confidence < min_conf)
        ]

        next_round = state.skeptic_rounds + 1
        if not challenged:
            critique = SkepticCritique(
                round=next_round,
                summary="All extracted claims are supported with high confidence.",
                requires_revision=False,
            )
            await ctx.events.emit(
                EventType.PROGRESS,
                message="skeptic audit passed: all claims verified",
                challenged=0,
                requires_revision=False,
            )
            return state.model_copy(
                update={
                    "skeptic_rounds": next_round,
                    "skeptic_history": [*state.skeptic_history, critique],
                    "pending_queries": [],
                }
            )

        claim_lines = [
            f"[{c.id}] (verdict={c.verdict.value if c.verdict else 'unknown'}, "
            f"confidence={c.confidence if c.confidence is not None else 'n/a'}): {c.text} "
            f"(verifier note: {c.rationale or 'none'})"
            for c in challenged
        ]
        result = await ctx.llm.generate_structured(
            system=prompts.SKEPTIC_SYSTEM,
            prompt=prompts.SKEPTIC_USER.format(
                question=state.question, claims="\n".join(claim_lines)
            ),
            schema=SkepticOut,
            purpose="skeptic",
        )

        notes_by_id: dict[str, str] = {}
        issues: list[SkepticIssue] = []
        counter_queries: list[str] = []
        valid_ids = {c.id for c in challenged}
        for item in result.value.issues:
            cid = item.claim_id.strip().strip("[]")
            if cid not in valid_ids:
                continue
            issue_text = item.issue.strip()
            query = item.counter_query.strip()
            notes_by_id[cid] = issue_text
            issues.append(SkepticIssue(claim_id=cid, issue=issue_text, counter_query=query))
            if query and query not in state.executed_queries and query not in counter_queries:
                counter_queries.append(query)

        updated_claims: list[Claim] = [
            c.model_copy(update={"skeptic_notes": notes_by_id[c.id]}) if c.id in notes_by_id else c
            for c in state.claims
        ]

        has_hard_failure = any(
            c.verdict in (Verdict.REFUTED, Verdict.NOT_ENOUGH_INFO) for c in challenged
        )
        can_loop = state.skeptic_rounds < ctx.settings.max_skeptic_rounds
        requires_revision = bool(can_loop and has_hard_failure and (counter_queries or issues))

        critique = SkepticCritique(
            round=next_round,
            summary=result.value.summary.strip(),
            requires_revision=requires_revision,
            issues=issues,
            counter_queries=counter_queries[:4],
        )
        await ctx.events.emit(
            EventType.PROGRESS,
            message=(
                f"skeptic challenged {len(challenged)} claim(s); "
                + (
                    f"triggering revision round {next_round}"
                    if requires_revision
                    else "proceeding to final report"
                )
            ),
            challenged=len(challenged),
            requires_revision=requires_revision,
            counter_queries=critique.counter_queries,
        )
        return state.model_copy(
            update={
                "claims": updated_claims,
                "skeptic_rounds": next_round,
                "skeptic_history": [*state.skeptic_history, critique],
                "pending_queries": critique.counter_queries if requires_revision else [],
            }
        )
