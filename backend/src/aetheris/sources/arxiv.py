"""arXiv API client (Atom feed). https://info.arxiv.org/help/api/user-manual.html"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from aetheris.core.models import Paper
from aetheris.retrieval.text import tokenize
from aetheris.sources.base import RateLimiter, get_with_backoff

API_URL = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_VERSION_RE = re.compile(r"v\d+$")


def build_query(query: str, *, max_terms: int = 6) -> str:
    """Turn a free-text query into an arXiv boolean query over all fields."""
    terms = list(dict.fromkeys(tokenize(query)))[:max_terms]
    if not terms:
        return f'all:"{query.strip()}"'
    return " AND ".join(f"all:{t}" for t in terms)


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", _NS):
        raw_id = (entry.findtext("a:id", default="", namespaces=_NS) or "").strip()
        if "/abs/" not in raw_id:
            continue  # error entries carry no abs link
        arxiv_id = _VERSION_RE.sub("", raw_id.rsplit("/abs/", 1)[1])
        title = _clean(entry.findtext("a:title", default="", namespaces=_NS))
        abstract = _clean(entry.findtext("a:summary", default="", namespaces=_NS))
        if not title or not abstract:
            continue
        published = entry.findtext("a:published", default="", namespaces=_NS) or ""
        authors = tuple(
            _clean(a.findtext("a:name", default="", namespaces=_NS))
            for a in entry.findall("a:author", _NS)
        )
        papers.append(
            Paper(
                id=f"arxiv:{arxiv_id}",
                source="arxiv",
                title=title,
                abstract=abstract,
                authors=tuple(a for a in authors if a),
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=_clean(entry.findtext("arxiv:journal_ref", default="", namespaces=_NS))
                or "arXiv",
                doi=(entry.findtext("arxiv:doi", default="", namespaces=_NS) or "").strip().lower()
                or None,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


class ArxivSource:
    name = "arxiv"

    def __init__(self, client: httpx.AsyncClient, *, limiter: RateLimiter | None = None) -> None:
        self._client = client
        # arXiv asks for no more than one request every three seconds.
        self._limiter = limiter or RateLimiter(3.0)

    async def search(self, query: str, *, limit: int) -> list[Paper]:
        response = await get_with_backoff(
            self._client,
            API_URL,
            params={
                "search_query": build_query(query),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            },
            limiter=self._limiter,
        )
        return parse_feed(response.text)
