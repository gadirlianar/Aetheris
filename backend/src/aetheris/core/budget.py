"""Hard resource limits for a single run."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from aetheris.core.errors import BudgetExceededError


@dataclass
class Budget:
    """Tracks LLM calls, tokens and wall-clock time; raises once any limit is crossed.

    Agents never check the budget themselves: the LLM wrapper charges it on every
    call and the engine checks the clock before every node.
    """

    max_llm_calls: int = 200
    max_total_tokens: int = 2_000_000
    max_seconds: float = 900.0
    clock: Callable[[], float] = time.monotonic

    llm_calls: int = 0
    total_tokens: int = 0
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started_at

    def check(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceededError(f"LLM call limit reached ({self.max_llm_calls})")
        if self.total_tokens >= self.max_total_tokens:
            raise BudgetExceededError(f"token limit reached ({self.max_total_tokens})")
        if self.elapsed >= self.max_seconds:
            raise BudgetExceededError(f"time limit reached ({self.max_seconds:.0f}s)")

    def charge(self, tokens: int) -> None:
        self.llm_calls += 1
        self.total_tokens += tokens
