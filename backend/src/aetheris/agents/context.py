"""Dependencies shared by every agent during a run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aetheris.core.config import Settings
from aetheris.core.events import EventBus
from aetheris.llm.instrumented import InstrumentedLLM
from aetheris.retrieval.store import EvidenceStore
from aetheris.sources.base import Source


@dataclass
class ResearchContext:
    settings: Settings
    llm: InstrumentedLLM
    evidence: EvidenceStore
    sources: Sequence[Source]
    events: EventBus
