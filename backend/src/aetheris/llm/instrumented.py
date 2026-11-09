"""Observability + budget enforcement around any :class:`LLMClient` / :class:`Embedder`."""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel

from aetheris.core.budget import Budget
from aetheris.core.events import EventBus, EventType
from aetheris.llm.base import Embedder, LLMClient, StructuredResult, Vector


class InstrumentedLLM:
    """Charges the budget, measures latency and emits an ``llm_call`` event per request."""

    def __init__(self, inner: LLMClient, events: EventBus, budget: Budget) -> None:
        self._inner = inner
        self._events = events
        self._budget = budget

    @property
    def model(self) -> str:
        return self._inner.model

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], purpose: str = ""
    ) -> StructuredResult[T]:
        self._budget.check()
        started = time.perf_counter()
        try:
            result = await self._inner.generate_structured(
                system=system, prompt=prompt, schema=schema
            )
        except Exception as exc:
            await self._events.emit(
                EventType.LLM_CALL,
                model=self.model,
                purpose=purpose,
                schema=schema.__name__,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:500],
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            raise
        self._budget.charge(result.usage.total)
        await self._events.emit(
            EventType.LLM_CALL,
            model=result.model,
            purpose=purpose,
            schema=schema.__name__,
            ok=True,
            attempts=result.attempts,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return result


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
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
