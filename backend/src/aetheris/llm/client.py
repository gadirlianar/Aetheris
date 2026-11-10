"""Tiered model routing and async token-bucket rate limiting for LLM execution.

Why tiered routing matters:
  * Query decomposition (`plan`) and atomic claim splitting (`extract_claims`) are
    structural transformations with low reasoning depth: routing them to a lightweight
    model (`fast_model`, e.g., `gemini-3.5-flash-lite`) cuts latency and token cost by 3x.
  * Evidence-grounded synthesis (`synthesize`), claim entailment (`verify`), adversarial
    critique (`skeptic`), and consensus/contradiction resolution (`contradiction`) demand
    high-precision multi-hop reasoning and are routed to `reasoning_model` (e.g.,
    `gemini-3.8-flash` or `gemini-4-argon`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from aetheris.core.budget import estimate_cost_usd
from aetheris.llm.base import LLMClient, ModelTier, StructuredResult

PURPOSE_TIER_MAP: dict[str, ModelTier] = {
    "plan": "fast",
    "extract_claims": "fast",
    "synthesize": "reasoning",
    "verify": "reasoning",
    "skeptic": "reasoning",
    "contradiction": "reasoning",
}


def resolve_tier(purpose: str = "", explicit_tier: ModelTier | None = None) -> ModelTier:
    if explicit_tier is not None:
        return explicit_tier
    return PURPOSE_TIER_MAP.get(purpose, "reasoning")


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter protecting concurrent LLM / verification bursts."""

    def __init__(
        self,
        rate_per_sec: float = 25.0,
        capacity: int = 10,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], asyncio.Future[None] | object] = asyncio.sleep,
    ) -> None:
        self._rate = max(0.1, float(rate_per_sec))
        self._capacity = max(1.0, float(capacity))
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._last_refill = self._clock()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> float:
        """Acquire ``amount`` tokens, sleeping if necessary. Returns wait time in seconds."""
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._last_refill)
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                deficit = amount - self._tokens
                delay = deficit / self._rate
            await self._sleep(delay)  # type: ignore[misc]
            waited += delay


@dataclass(frozen=True)
class ModelAliasClient:
    """Thin wrapper overriding the reported model name on an existing ``LLMClient``."""

    inner: LLMClient
    alias_model: str

    @property
    def model(self) -> str:
        return self.alias_model

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        res = await self.inner.generate_structured(system=system, prompt=prompt, schema=schema)
        return StructuredResult(
            value=res.value,
            usage=res.usage,
            model=self.alias_model,
            attempts=res.attempts,
            tier=res.tier,
            cost_usd=res.cost_usd,
        )


class TieredLLMRouter:
    """Routes structured generation calls between a fast tier and a deep reasoning tier."""

    def __init__(
        self,
        *,
        fast_client: LLMClient,
        reasoning_client: LLMClient,
        purpose_map: Mapping[str, ModelTier] | None = None,
    ) -> None:
        self._fast = fast_client
        self._reasoning = reasoning_client
        self._purpose_map = dict(PURPOSE_TIER_MAP)
        if purpose_map:
            self._purpose_map.update(purpose_map)

    @property
    def model(self) -> str:
        return f"tiered({self._fast.model} -> {self._reasoning.model})"

    @property
    def fast_model(self) -> str:
        return self._fast.model

    @property
    def reasoning_model(self) -> str:
        return self._reasoning.model

    def select(
        self, *, purpose: str = "", tier: ModelTier | None = None
    ) -> tuple[LLMClient, ModelTier]:
        resolved: ModelTier = (
            tier if tier is not None else self._purpose_map.get(purpose, "reasoning")
        )
        client = self._fast if resolved == "fast" else self._reasoning
        return client, resolved

    async def generate_structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str = "",
        tier: ModelTier | None = None,
    ) -> StructuredResult[T]:
        client, resolved_tier = self.select(purpose=purpose, tier=tier)
        raw = await client.generate_structured(system=system, prompt=prompt, schema=schema)
        cost = estimate_cost_usd(
            raw.model, raw.usage.input_tokens, raw.usage.output_tokens, tier=resolved_tier
        )
        return StructuredResult(
            value=raw.value,
            usage=raw.usage,
            model=raw.model,
            attempts=raw.attempts,
            tier=resolved_tier,
            cost_usd=cost,
        )

