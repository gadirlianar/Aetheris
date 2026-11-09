"""Rendering evidence for prompts with short, model-friendly aliases (E1, E2, ...).

Short aliases instead of raw chunk ids cut tokens and make citation hallucination
detectable: anything that is not a known alias is, by construction, invalid.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from aetheris.core.models import Chunk, Paper

_LABEL_RE = re.compile(r"[A-Za-z]+\d+")


@dataclass
class EvidenceBlock:
    text: str
    alias_to_chunk: dict[str, str] = field(default_factory=dict)

    def resolve(self, label: str) -> str | None:
        match = _LABEL_RE.search(label)
        return self.alias_to_chunk.get(match.group(0).upper()) if match else None


def format_evidence(
    chunks: Sequence[Chunk], papers: Mapping[str, Paper], *, prefix: str = "E"
) -> EvidenceBlock:
    lines: list[str] = []
    aliases: dict[str, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        alias = f"{prefix}{i}"
        aliases[alias] = chunk.id
        paper = papers.get(chunk.paper_id)
        header = f'({paper.short_citation()}, "{paper.title}")' if paper else ""
        lines.append(f"[{alias}] {header}\n{chunk.text}")
    return EvidenceBlock("\n\n".join(lines), aliases)
