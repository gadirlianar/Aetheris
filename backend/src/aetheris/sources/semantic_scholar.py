"""Semantic Scholar Academic Graph API v1 client.

Reference: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations

from typing import Any

import httpx

from aetheris.core.models import Paper
from aetheris.sources.base import RateLimiter, get_with_backoff

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,year,venue,authors,externalIds,url,citationCount"


def parse_paper(item: dict[str, Any]) -> Paper | None:
    paper_id = (item.get("paperId") or "").strip()
    title = " ".join((item.get("title") or "").split())
    abstract = " ".join((item.get("abstract") or "").split())
    if not (paper_id and title and abstract):
        return None
    ext = item.get("externalIds") or {}
    doi = (ext.get("DOI") or "").strip().lower() or None
    arxiv_id = (ext.get("ArXiv") or "").strip()
    url = (
        item.get("url")
        or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
        or (
            f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{paper_id}"
        )
    )
    authors = tuple(
        a["name"].strip()
        for a in (item.get("authors") or [])
        if isinstance(a, dict) and (a.get("name") or "").strip()
    )
    return Paper(
        id=f"s2:{paper_id}",
        source="semantic_scholar",
        title=title,
        abstract=abstract,
        authors=authors,
        year=item.get("year"),
        venue=(item.get("venue") or "").strip() or None,
        doi=doi,
        url=url,
        citation_count=item.get("citationCount"),
    )


class SemanticScholarSource:
    name = "semantic_scholar"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        # Unauthenticated limit is 1 req/s; polite interval 1.1s.
        self._limiter = limiter or RateLimiter(1.1)

    async def search(self, query: str, *, limit: int) -> list[Paper]:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 25),
            "fields": _FIELDS,
        }
        if self._api_key:
            self._client.headers["x-api-key"] = self._api_key
        response = await get_with_backoff(
            self._client, API_URL, params=params, limiter=self._limiter
        )
        data = response.json().get("data") or []
        return [p for p in (parse_paper(row) for row in data) if p is not None]
