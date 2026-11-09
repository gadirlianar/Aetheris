"""Gemini adapters for :class:`LLMClient` and :class:`Embedder` (Interactions API)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from aetheris.core.errors import LLMOutputError, TransientError
from aetheris.llm.base import StructuredResult, TokenUsage, Vector, l2_normalize

_TRANSIENT_CODES = frozenset({408, 429, 500, 502, 503, 504})

_REPAIR_SUFFIX = (
    "\n\n---\nYour previous answer could not be parsed against the required JSON schema.\n"
    "Validation error:\n{error}\n\nReturn ONLY a corrected JSON object."
)


def _translate(exc: genai_errors.APIError) -> Exception:
    if exc.code in _TRANSIENT_CODES:
        return TransientError(f"Gemini API {exc.code}: {exc}")
    return exc


class GeminiLLM:
    """Structured generation with schema-constrained decoding plus one self-repair round."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        thinking_level: str | None = "low",
        max_repair_attempts: int = 1,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._thinking_level = thinking_level
        self._max_repair_attempts = max_repair_attempts

    @property
    def model(self) -> str:
        return self._model

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        usage = TokenUsage()
        current_prompt = prompt
        last_error: ValidationError | None = None
        for attempt in range(1, self._max_repair_attempts + 2):
            request: dict[str, Any] = {
                "model": self._model,
                "input": current_prompt,
                "system_instruction": system,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                },
                "store": False,
            }
            if self._thinking_level:
                request["generation_config"] = {"thinking_level": self._thinking_level}
            try:
                interaction = await self._client.aio.interactions.create(**request)
            except genai_errors.APIError as exc:
                raise _translate(exc) from exc

            if interaction.usage is not None:
                usage.input_tokens += interaction.usage.total_input_tokens or 0
                usage.output_tokens += (interaction.usage.total_output_tokens or 0) + (
                    interaction.usage.total_thought_tokens or 0
                )
            text = interaction.output_text or ""
            try:
                value = schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = exc
                current_prompt = prompt + _REPAIR_SUFFIX.format(error=str(exc)[:2000])
                continue
            return StructuredResult(value=value, usage=usage, model=self._model, attempts=attempt)

        raise LLMOutputError(
            f"{schema.__name__}: output failed validation after "
            f"{self._max_repair_attempts + 1} attempts: {last_error}"
        )


class GeminiEmbedder:
    """``gemini-embedding-2`` with the task-prefix format recommended for asymmetric retrieval."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        *,
        dim: int = 768,
        batch_size: int = 100,
        concurrency: int = 4,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> Vector:
        formatted = [f"title: {title or 'none'} | text: {text}" for title, text in docs]
        return await self._embed(formatted)

    async def embed_queries(self, queries: Sequence[str]) -> Vector:
        return await self._embed([f"task: search result | query: {q}" for q in queries])

    async def _embed(self, texts: list[str]) -> Vector:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        return l2_normalize(np.vstack(results))

    async def _embed_batch(self, batch: list[str]) -> Vector:
        async with self._semaphore:
            try:
                response = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(output_dimensionality=self._dim),
                )
            except genai_errors.APIError as exc:
                raise _translate(exc) from exc
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise TransientError(f"expected {len(batch)} embeddings, got {len(embeddings)}")
        return np.asarray([e.values for e in embeddings], dtype=np.float32)
