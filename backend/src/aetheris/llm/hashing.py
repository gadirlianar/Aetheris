"""Deterministic subword + n-gram signed random-projection embedder.

Used when running offline (`--demo` or `aetheris eval` without `GEMINI_API_KEY`).
Unlike a naive word-hashing stub, this implementation combines:
  1. Sublinear TF-weighted word unigrams (`1 + log(tf)`),
  2. Word bigrams (`w_i + "_" + w_{i+1}`) for local phrase order, and
  3. Character 4-grams (`#c0c1c2c3#`) for morphological and cross-terminology
     similarity (so e.g. `"hallucinate"` and `"hallucinations"`, or `"ribonucleoprotein"`
     and `"ribonuclease"` share dense subword geometry even when exact BM25 stems differ).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
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
            tf_counts = Counter(tokens)
            weighted_features: list[tuple[str, float]] = []

            # 1. Sublinear TF word unigrams
            for tok, count in tf_counts.items():
                weighted_features.append((f"u:{tok}", 1.0 + math.log(count)))
                # 3. Character 4-grams for subword/morphological representation
                padded = f"<{tok}>"
                if len(padded) >= 4:
                    for idx in range(len(padded) - 3):
                        weighted_features.append((f"c:{padded[idx : idx + 4]}", 0.35))

            # 2. Word bigrams
            for a, b in itertools.pairwise(tokens):
                weighted_features.append((f"b:{a}_{b}", 0.85))

            for feature, weight in weighted_features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                matrix[row, h % self._dim] += sign * weight
        return l2_normalize(matrix)
