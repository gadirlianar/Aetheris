"""Deterministic, network-free test doubles."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from aetheris.llm.base import StructuredResult, TokenUsage

Handler = Callable[[str, str], BaseModel]
"""``(system, prompt) -> instance of the requested schema``."""


@dataclass
class FakeCall:
    system: str
    prompt: str
    schema: type[BaseModel]


@dataclass
class FakeLLM:
    """Routes each request to a handler keyed by the requested output schema."""

    handlers: Mapping[type[BaseModel], Handler]
    calls: list[FakeCall] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "fake"

    async def generate_structured[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        self.calls.append(FakeCall(system, prompt, schema))
        try:
            handler = self.handlers[schema]
        except KeyError:
            raise AssertionError(f"FakeLLM has no handler for {schema.__name__}") from None
        produced = handler(system, prompt)
        # Round-trip through JSON exactly like a real provider would.
        value = schema.model_validate_json(produced.model_dump_json())
        usage = TokenUsage(
            input_tokens=(len(system) + len(prompt)) // 4,
            output_tokens=len(json.dumps(produced.model_dump())) // 4,
        )
        return StructuredResult(value=value, usage=usage, model=self.model)

    def calls_for(self, schema: type[BaseModel]) -> list[FakeCall]:
        return [c for c in self.calls if c.schema is schema]
