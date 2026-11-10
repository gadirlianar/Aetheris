"""Observability, tiered routing, rate-limiting & budget enforcement around LLM / Embedder."""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel

from aetheris.core.budget import Budget
from aetheris.core.events import EventBus, EventType, current_node
from aetheris.llm.base import Embedder, LLMClient, ModelTier, StructuredResult, Vector
from aetheris.llm.client import TieredLLMRouter, TokenBucketRateLimiter, resolve_tier


class InstrumentedLLM:
    """Charges the budget, enforces rate limits, routes model tiers, and emits ``llm_call``."""

    def __init__(
        self,
        inner: LLMClient | TieredLLMRouter,
        events: EventBus,
        budget: Budget,
        *,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self._inner = inner
        self._events = events
        self._budget = budget
        self._rate_limiter = rate_limiter

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def budget(self) -> Budget:
        return self._budget

    async def generate_structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str = "",
        tier: ModelTier | None = None,
    ) -> StructuredResult[T]:
        self._budget.check()
        resolved_tier = resolve_tier(purpose, tier)
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(1.0)

        started = time.perf_counter()
        try:
            if isinstance(self._inner, TieredLLMRouter):
                result = await self._inner.generate_structured(
                    system=system,
                    prompt=prompt,
                    schema=schema,
                    purpose=purpose,
                    tier=resolved_tier,
                )
            else:
                result = await self._inner.generate_structured(
                    system=system, prompt=prompt, schema=schema
                )
        except Exception as exc:
            await self._events.emit(
                EventType.LLM_CALL,
                model=self.model,
                tier=resolved_tier,
                purpose=purpose,
                schema=schema.__name__,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:500],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        active_node = current_node.get() or purpose or "llm"
        call_cost = self._budget.charge(
            result.usage.total,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            model=result.model,
            tier=resolved_tier,
            node=active_node,
            latency_ms=latency_ms,
        )
        enriched = StructuredResult(
            value=result.value,
            usage=result.usage,
            model=result.model,
            attempts=result.attempts,
            tier=resolved_tier,
            cost_usd=call_cost,
        )
        await self._events.emit(
            EventType.LLM_CALL,
            model=enriched.model,
            tier=resolved_tier,
            purpose=purpose,
            schema=schema.__name__,
            ok=True,
            attempts=enriched.attempts,
            input_tokens=enriched.usage.input_tokens,
            output_tokens=enriched.usage.output_tokens,
            cost_usd=call_cost,
            latency_ms=latency_ms,
        )
        return enriched


class InstrumentedEmbedder:
    def __init__(self, inner: Embedder, events: EventBus) -> None:
        self._inner = inner
        self._events = events

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> Vector:
        started = time.perf_counter()
        vectors = await self._inner.embed_documents(docs)
        await self._emit("documents", len(docs), started)
        return vectors

    async def embed_queries(self, queries: Sequence[str]) -> Vector:
        started = time.perf_counter()
        vectors = await self._inner.embed_queries(queries)
        await self._emit("queries", len(queries), started)
        return vectors

    async def _emit(self, kind: str, n: int, started: float) -> None:
        await self._events.emit(
            EventType.EMBEDDING_CALL,
            kind=kind,
            n=n,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
