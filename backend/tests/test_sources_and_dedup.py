"""Unit tests for arXiv, OpenAlex, Semantic Scholar HTTP clients, backoff, and dedup."""

from __future__ import annotations

import httpx
import respx

from aetheris.core.models import Paper
from aetheris.sources import arxiv, openalex, semantic_scholar
from aetheris.sources.base import RateLimiter
from aetheris.sources.dedup import merge_papers

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:00:00Z</published>
    <title>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</title>
    <summary>We endow pre-trained models with non-parametric memory.</summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Douwe Kiela</name></author>
    <arxiv:doi>10.48550/arXiv.2005.11401</arxiv:doi>
  </entry>
</feed>
"""


@respx.mock
async def test_arxiv_openalex_and_s2_parsers_and_backoff() -> None:
    # First arXiv attempt returns 429 with Retry-After, second succeeds
    route_arxiv = respx.get(arxiv.API_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0.01"}),
            httpx.Response(200, text=SAMPLE_ARXIV_XML),
        ]
    )
    respx.get(openalex.API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": (
                            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
                        ),
                        "publication_year": 2020,
                        "doi": "https://doi.org/10.48550/arxiv.2005.11401",
                        "cited_by_count": 6840,
                        "primary_location": {
                            "landing_page_url": "https://arxiv.org/abs/2005.11401",
                            "source": {"display_name": "NeurIPS"},
                        },
                        "authorships": [{"author": {"display_name": "Patrick Lewis"}}],
                        "abstract_inverted_index": {
                            "We": [0],
                            "endow": [1],
                            "models.": [2],
                        },
                    }
                ]
            },
        )
    )
    respx.get(semantic_scholar.API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "s2abc",
                        "title": "GraphRAG Multi-Hop Synthesis",
                        "abstract": "GraphRAG extracts an entity graph from scientific corpora.",
                        "year": 2024,
                        "venue": "ACL",
                        "citationCount": 250,
                        "authors": [{"name": "Darren Edge"}],
                        "externalIds": {"DOI": "10.1000/graphrag"},
                    }
                ]
            },
        )
    )

    fast_limiter = RateLimiter(0.0)
    async with httpx.AsyncClient() as client:
        ax = await arxiv.ArxivSource(client, limiter=fast_limiter).search("rag", limit=5)
        oa = await openalex.OpenAlexSource(
            client, email="test@example.com", limiter=fast_limiter
        ).search("rag", limit=5)
        s2 = await semantic_scholar.SemanticScholarSource(
            client, api_key="k", limiter=fast_limiter
        ).search("graphrag", limit=5)

    assert route_arxiv.call_count == 2
    assert len(ax) == 1
    assert ax[0].id == "arxiv:2005.11401"
    assert len(oa) == 1
    assert oa[0].venue == "NeurIPS"
    assert len(s2) == 1
    assert s2[0].id == "s2:s2abc"

    # Cross-source dedup merges oa metadata (venue="NeurIPS", citation_count=6840) into ax[0]
    catalog: dict[str, Paper] = {}
    added_first = merge_papers(catalog, ax)
    added_second = merge_papers(catalog, oa)
    added_third = merge_papers(catalog, s2)
    assert len(added_first) == 1
    assert len(added_second) == 0
    assert len(added_third) == 1
    merged = catalog["arxiv:2005.11401"]
    assert merged.venue == "NeurIPS"
    assert merged.citation_count == 6840
