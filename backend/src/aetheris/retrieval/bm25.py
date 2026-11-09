"""Okapi BM25 over pre-tokenised documents."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class BM25Index:
    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf = [Counter(doc) for doc in documents]
        self._len = np.asarray([len(doc) for doc in documents], dtype=np.float64)
        self._avg_len = float(self._len.mean()) if len(documents) else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documents)
        # BM25+ style idf floor keeps very common terms from going negative.
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def scores(self, query: Sequence[str]) -> NDArray[np.float64]:
        out = np.zeros(len(self._tf), dtype=np.float64)
        if not self._tf or self._avg_len == 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self._len / self._avg_len)
        for term in set(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.asarray([doc.get(term, 0) for doc in self._tf], dtype=np.float64)
            out += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out
