"""Hard resource limits, USD cost accounting, and latency percentile telemetry."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from aetheris.core.errors import BudgetExceededError

# Per-1M token pricing in USD: (input_usd_per_1m, output_usd_per_1m)
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.075, 0.30),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.8-flash": (0.30, 1.20),
    "gemini-4-argon": (1.25, 5.00),
    "aetheris-demo-fast": (0.10, 0.40),
    "aetheris-demo-reasoning": (0.30, 1.20),
    "aetheris-demo-1.0": (0.30, 1.20),
    "aetheris-benchmark-verifier": (0.30, 1.20),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    tier: str = "reasoning",
) -> float:
    """Compute exact USD cost from token counts and model/tier pricing."""
    default_rates = (0.10, 0.40) if tier == "fast" else (0.30, 1.20)
    in_rate, out_rate = MODEL_PRICING_PER_1M_TOKENS.get(model, default_rates)
    cost = (max(0, input_tokens) * in_rate + max(0, output_tokens) * out_rate) / 1_000_000.0
    return round(cost, 8)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(float(sorted_vals[0]), 2)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return round(float(sorted_vals[lower]), 2)
    weight = pos - lower
    val = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
    return round(float(val), 2)


@dataclass(frozen=True)
class LatencyPercentiles:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_latency_percentiles(samples_ms: Sequence[float]) -> LatencyPercentiles:
    if not samples_ms:
        return LatencyPercentiles()
    ordered = sorted(float(x) for x in samples_ms)
    n = len(ordered)
    mean_val = round(sum(ordered) / n, 2)
    return LatencyPercentiles(
        count=n,
        mean_ms=mean_val,
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=_percentile(ordered, 95.0),
        p99_ms=_percentile(ordered, 99.0),
        max_ms=round(ordered[-1], 2),
    )


@dataclass
class TierUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Budget:
    """Tracks LLM calls, tokens, USD cost, per-tier routing, and per-node latency percentiles.

    Agents never check the budget themselves: the LLM wrapper charges it on every
    call and the engine checks the clock and cost before every node.
    """

    max_llm_calls: int = 200
    max_total_tokens: int = 2_000_000
    max_seconds: float = 900.0
    max_cost_usd: float = 5.0
    clock: Callable[[], float] = time.monotonic

    llm_calls: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    started_at: float = field(init=False)
    tier_usage: dict[str, TierUsage] = field(
        default_factory=lambda: {"fast": TierUsage(), "reasoning": TierUsage()}
    )
    node_latencies_ms: dict[str, list[float]] = field(default_factory=dict)
    all_latencies_ms: list[float] = field(default_factory=list)

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
        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(f"USD cost limit reached (${self.max_cost_usd:.4f})")
        if self.elapsed >= self.max_seconds:
            raise BudgetExceededError(f"time limit reached ({self.max_seconds:.0f}s)")

    def charge(
        self,
        tokens: int,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str = "",
        tier: str = "reasoning",
        node: str | None = None,
        latency_ms: float | None = None,
    ) -> float:
        self.llm_calls += 1
        self.total_tokens += tokens
        in_tok = input_tokens if input_tokens is not None else int(tokens * 0.7)
        out_tok = output_tokens if output_tokens is not None else (tokens - in_tok)
        self.input_tokens += in_tok
        self.output_tokens += out_tok

        call_cost = estimate_cost_usd(model, in_tok, out_tok, tier=tier)
        self.total_cost_usd = round(self.total_cost_usd + call_cost, 8)

        bucket = self.tier_usage.setdefault(tier, TierUsage())
        bucket.calls += 1
        bucket.input_tokens += in_tok
        bucket.output_tokens += out_tok
        bucket.cost_usd = round(bucket.cost_usd + call_cost, 8)

        if latency_ms is not None:
            self.all_latencies_ms.append(float(latency_ms))
            if node:
                self.node_latencies_ms.setdefault(node, []).append(float(latency_ms))
        return call_cost

    def record_node_latency(self, node: str, duration_ms: float) -> None:
        self.node_latencies_ms.setdefault(node, []).append(float(duration_ms))

    def latency_percentiles(self, node: str | None = None) -> LatencyPercentiles:
        if node is not None:
            return compute_latency_percentiles(self.node_latencies_ms.get(node, ()))
        return compute_latency_percentiles(self.all_latencies_ms)

    def telemetry_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "tier_usage": {k: v.to_dict() for k, v in self.tier_usage.items()},
            "llm_latency": self.latency_percentiles().to_dict(),
            "node_latencies": {
                node: compute_latency_percentiles(samples).to_dict()
                for node, samples in self.node_latencies_ms.items()
            },
        }
