"""Provider-agnostic LLM and embedding interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

Vector = NDArray[np.float32]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    usage: TokenUsage
    model: str
    attempts: int = 1


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated Pydantic object."""

    @property
    def model(self) -> str: ...

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]: ...


class Embedder(Protocol):
    """Produces L2-normalised embeddings (rows of a 2-D float32 array)."""

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> Vector:
        """Embed ``(title, text)`` pairs."""
        ...

    async def embed_queries(self, queries: Sequence[str]) -> Vector: ...


def l2_normalize(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)
