"""OpenAlex works search. https://docs.openalex.org/"""

from __future__ import annotations

from typing import Any

import httpx

from aetheris.core.models import Paper
from aetheris.sources.base import RateLimiter, get_with_backoff

API_URL = "https://api.openalex.org/works"
_FIELDS = (
    "id,display_name,publication_year,authorships,primary_location,doi,"
    "cited_by_count,abstract_inverted_index"
)


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as ``{word: [positions]}``; rebuild the text."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def parse_work(work: dict[str, Any]) -> Paper | None:
    title = " ".join((work.get("display_name") or "").split())
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    work_id = (work.get("id") or "").rsplit("/", 1)[-1]
    if not (title and abstract and work_id):
        return None
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    doi = (work.get("doi") or "").removeprefix("https://doi.org/").lower() or None
    url = location.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else work["id"])
    authors = tuple(
        a["author"]["display_name"]
        for a in work.get("authorships") or []
        if (a.get("author") or {}).get("display_name")
    )
    return Paper(
        id=f"openalex:{work_id}",
        source="openalex",
        title=title,
        abstract=abstract,
        authors=authors,
        year=work.get("publication_year"),
        venue=source.get("display_name"),
        doi=doi,
        url=url,
        citation_count=work.get("cited_by_count"),
    )


class OpenAlexSource:
    name = "openalex"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        email: str | None = None,
        api_key: str | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._client = client
        self._email = email
        self._api_key = api_key
        self._limiter = limiter or RateLimiter(0.15)

    async def search(self, query: str, *, limit: int) -> list[Paper]:
        params: dict[str, Any] = {
            "search": query,
            "filter": "has_abstract:true",
            "per-page": limit,
            "select": _FIELDS,
        }
        if self._email:
            params["mailto"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        response = await get_with_backoff(
            self._client, API_URL, params=params, limiter=self._limiter
        )
        results = response.json().get("results") or []
        return [p for p in (parse_work(w) for w in results) if p is not None]
