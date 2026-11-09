"""Cross-source de-duplication (the same paper often lives on arXiv *and* OpenAlex)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from aetheris.core.models import Paper

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def title_key(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower())


def merge_papers(existing: dict[str, Paper], incoming: Iterable[Paper]) -> list[Paper]:
    """Add ``incoming`` to ``existing`` in place; return only genuinely new papers.

    Identity is decided by id, DOI, or normalised title. When a duplicate is
    found the first-seen record wins, but missing metadata (citation count,
    DOI, venue) is back-filled from the duplicate.
    """
    by_doi = {p.doi: pid for pid, p in existing.items() if p.doi}
    by_title = {title_key(p.title): pid for pid, p in existing.items()}
    added: list[Paper] = []
    for paper in incoming:
        match = (
            paper.id
            if paper.id in existing
            else by_doi.get(paper.doi or "") or by_title.get(title_key(paper.title))
        )
        if match is None:
            existing[paper.id] = paper
            if paper.doi:
                by_doi[paper.doi] = paper.id
            by_title[title_key(paper.title)] = paper.id
            added.append(paper)
            continue
        current = existing[match]
        updates = {
            name: getattr(paper, name)
            for name in ("doi", "venue", "citation_count", "year")
            if getattr(current, name) in (None, "", "arXiv") and getattr(paper, name)
        }
        if updates:
            existing[match] = current.model_copy(update=updates)
    return added
