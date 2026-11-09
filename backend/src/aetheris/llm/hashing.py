"""A deterministic feature-hashing embedder.

Used in tests and as an offline fallback. It captures lexical and bigram overlap,
which makes test expectations about ranking deterministic and fast.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np

from aetheris.llm.base import Vector, l2_normalize
from aetheris.retrieval.text import tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> Vector:
        return self._embed([f"{title} {text}" for title, text in docs])

    async def embed_queries(self, queries: Sequence[str]) -> Vector:
        return self._embed(list(queries))

    def _embed(self, texts: list[str]) -> Vector:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                matrix[row, h % self._dim] += sign
        return l2_normalize(matrix)
