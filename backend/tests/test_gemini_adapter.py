"""Unit tests for GeminiLLM (including schema self-repair loop) and GeminiEmbedder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from aetheris.core.errors import LLMOutputError, TransientError
from aetheris.llm.fake import FakeLLM
from aetheris.llm.gemini import GeminiEmbedder, GeminiLLM


class TinySchema(BaseModel):
    answer: str
    score: int


@dataclass
class _MockUsage:
    total_input_tokens: int = 12
    total_output_tokens: int = 8
    total_thought_tokens: int = 4


@dataclass
class _MockInteraction:
    output_text: str
    usage: _MockUsage


@dataclass
class _MockEmbedding:
    values: list[float]


@dataclass
class _MockEmbedResponse:
    embeddings: list[_MockEmbedding]


class _MockAioInteractions:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _MockInteraction:
        self.requests.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _MockAioModels:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        del model, config
        return _MockEmbedResponse(embeddings=[_MockEmbedding([1.0] * self._dim) for _ in contents])


class _MockAio:
    def __init__(self, responses: list[Any], dim: int = 4) -> None:
        self.interactions = _MockAioInteractions(responses)
        self.models = _MockAioModels(dim=dim)


class _MockGenAIClient:
    def __init__(self, responses: list[Any], dim: int = 4) -> None:
        self.aio = _MockAio(responses, dim=dim)


async def test_gemini_llm_self_repair_and_error_translation() -> None:
    # First response is invalid JSON, second is valid -> self-repair succeeds on attempt 2
    mock_client = _MockGenAIClient(
        [
            _MockInteraction(
                output_text='{"answer": "ok", "score": "not-an-int"}', usage=_MockUsage()
            ),
            _MockInteraction(output_text='{"answer": "fixed", "score": 42}', usage=_MockUsage()),
        ]
    )
    llm = GeminiLLM(
        "fake-key",
        "gemini-3.8-flash",
        max_repair_attempts=1,
        client=mock_client,  # type: ignore[arg-type]
    )
    res = await llm.generate_structured(system="sys", prompt="user", schema=TinySchema)
    assert res.value.answer == "fixed"
    assert res.value.score == 42
    assert res.attempts == 2
    assert res.usage.input_tokens == 24
    assert res.usage.output_tokens == 24

    # Exhausting repair attempts raises LLMOutputError
    bad_client = _MockGenAIClient(
        [
            _MockInteraction(output_text="not json", usage=_MockUsage()),
            _MockInteraction(output_text="still not json", usage=_MockUsage()),
        ]
    )
    bad_llm = GeminiLLM("k", "gemini-3.8-flash", max_repair_attempts=1, client=bad_client)  # type: ignore[arg-type]
    with pytest.raises(LLMOutputError):
        await bad_llm.generate_structured(system="s", prompt="p", schema=TinySchema)

    # HTTP 429 APIError is translated to TransientError
    rate_err = genai_errors.APIError(429, {"error": {"message": "quota"}})
    err_client = _MockGenAIClient([rate_err])
    err_llm = GeminiLLM("k", "gemini-3.8-flash", client=err_client)  # type: ignore[arg-type]
    with pytest.raises(TransientError, match="429"):
        await err_llm.generate_structured(system="s", prompt="p", schema=TinySchema)


async def test_gemini_embedder_and_fake_llm() -> None:
    mock_client = _MockGenAIClient([], dim=4)
    emb = GeminiEmbedder("k", dim=4, batch_size=2, client=mock_client)  # type: ignore[arg-type]
    doc_vecs = await emb.embed_documents([("T1", "B1"), ("T2", "B2"), ("T3", "B3")])
    assert doc_vecs.shape == (3, 4)
    q_vecs = await emb.embed_queries(["query 1"])
    assert q_vecs.shape == (1, 4)
    empty = await emb.embed_queries([])
    assert empty.shape == (0, 4)

    fake = FakeLLM({TinySchema: lambda _s, _p: TinySchema(answer="from_fake", score=7)})
    out = await fake.generate_structured(system="s", prompt="p", schema=TinySchema)
    assert out.value.answer == "from_fake"
    assert len(fake.calls_for(TinySchema)) == 1
