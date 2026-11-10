"""Planner: question -> research plan (sub-questions + search queries)."""

from __future__ import annotations

from aetheris.agents import prompts
from aetheris.agents.context import ResearchContext
from aetheris.core.events import EventType
from aetheris.core.models import ResearchPlan, ResearchState, SubQuestion


class Planner:
    async def run(self, state: ResearchState, ctx: ResearchContext) -> ResearchState:
        s = ctx.settings
        result = await ctx.llm.generate_structured(
            system=prompts.PLANNER_SYSTEM.format(
                max_sub_questions=s.max_sub_questions, max_queries=s.max_queries_per_sub_question
            ),
            prompt=prompts.PLANNER_USER.format(question=state.question),
            schema=ResearchPlan,
            purpose="plan",
            tier="fast",
        )
        plan = _enforce_bounds(result.value, s.max_sub_questions, s.max_queries_per_sub_question)
        await ctx.events.emit(
            EventType.PROGRESS,
            message=f"planned {len(plan.sub_questions)} sub-questions",
            sub_questions=[sq.question for sq in plan.sub_questions],
        )
        return state.model_copy(update={"plan": plan})


def _enforce_bounds(plan: ResearchPlan, max_sq: int, max_q: int) -> ResearchPlan:
    """Bounds live here, not in the JSON schema, to keep constrained decoding cheap."""
    sub_questions = []
    for sq in plan.sub_questions[:max_sq]:
        queries = [q.strip() for q in sq.search_queries if q.strip()][:max_q] or [sq.question]
        sub_questions.append(
            SubQuestion(
                question=sq.question.strip(),
                rationale=sq.rationale,
                search_queries=queries,
            )
        )
    if not sub_questions:
        raise ValueError("planner returned no sub-questions")
    return ResearchPlan(objective=plan.objective, sub_questions=sub_questions)
