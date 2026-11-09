"""Declarative execution graph.

A graph is a set of named nodes, each with exactly one way out: either a static
edge or a conditional router. ``validate()`` catches wiring mistakes (dangling
edges, unreachable nodes, nodes with no exit) before anything runs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, Self

from aetheris.core.errors import GraphValidationError, TransientError

END = "__end__"


class Node[S, C](Protocol):
    """A unit of work: takes the current state, returns the next state."""

    async def run(self, state: S, ctx: C) -> S: ...


Router = Callable[[object], str]


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.2
    retry_on: tuple[type[BaseException], ...] = (TransientError, TimeoutError)

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before attempt ``attempt + 1`` (``attempt`` is 1-based)."""
        base = min(self.max_delay, self.initial_delay * self.multiplier ** (attempt - 1))
        spread = base * self.jitter
        return max(0.0, base + (rng or random).uniform(-spread, spread))


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass
class NodeSpec[S, C]:
    name: str
    node: Node[S, C]
    retry: RetryPolicy = NO_RETRY
    timeout_s: float | None = None


@dataclass
class _Conditional:
    router: Callable[..., str]
    targets: frozenset[str]


@dataclass
class Graph[S, C]:
    nodes: dict[str, NodeSpec[S, C]] = field(default_factory=dict)
    entry: str | None = None
    _edges: dict[str, str] = field(default_factory=dict)
    _conditionals: dict[str, _Conditional] = field(default_factory=dict)

    # ------------------------------------------------------------------ building

    def add_node(
        self,
        name: str,
        node: Node[S, C],
        *,
        retry: RetryPolicy = NO_RETRY,
        timeout_s: float | None = None,
    ) -> Self:
        if name == END:
            raise GraphValidationError(f"'{END}' is reserved")
        if name in self.nodes:
            raise GraphValidationError(f"duplicate node '{name}'")
        self.nodes[name] = NodeSpec(name, node, retry, timeout_s)
        if self.entry is None:
            self.entry = name
        return self

    def add_edge(self, src: str, dst: str) -> Self:
        self._assert_no_exit(src)
        self._edges[src] = dst
        return self

    def add_conditional(self, src: str, router: Callable[[S], str], targets: Iterable[str]) -> Self:
        self._assert_no_exit(src)
        self._conditionals[src] = _Conditional(router, frozenset(targets))
        return self

    def set_entry(self, name: str) -> Self:
        self.entry = name
        return self

    def _assert_no_exit(self, src: str) -> None:
        if src in self._edges or src in self._conditionals:
            raise GraphValidationError(f"node '{src}' already has an outgoing transition")

    # ------------------------------------------------------------------ querying

    def successors(self, name: str) -> frozenset[str]:
        if name in self._edges:
            return frozenset({self._edges[name]})
        if name in self._conditionals:
            return self._conditionals[name].targets
        return frozenset()

    def next_node(self, name: str, state: S) -> str:
        if name in self._edges:
            return self._edges[name]
        conditional = self._conditionals[name]
        target = conditional.router(state)
        if target not in conditional.targets:
            raise GraphValidationError(
                f"router of '{name}' returned '{target}', expected one of "
                f"{sorted(conditional.targets)}"
            )
        return target

    def validate(self) -> None:
        if not self.nodes:
            raise GraphValidationError("graph has no nodes")
        if self.entry not in self.nodes:
            raise GraphValidationError(f"entry node '{self.entry}' does not exist")
        for name in self.nodes:
            succ = self.successors(name)
            if not succ:
                raise GraphValidationError(f"node '{name}' has no outgoing transition")
            unknown = {s for s in succ if s != END and s not in self.nodes}
            if unknown:
                raise GraphValidationError(f"node '{name}' points to unknown {sorted(unknown)}")
        for src in (*self._edges, *self._conditionals):
            if src not in self.nodes:
                raise GraphValidationError(f"transition from unknown node '{src}'")

        reachable: set[str] = set()
        frontier = [self.entry]
        while frontier:
            current = frontier.pop()
            if current in reachable or current == END:
                continue
            reachable.add(current)
            frontier.extend(self.successors(current))
        unreachable = set(self.nodes) - reachable
        if unreachable:
            raise GraphValidationError(f"unreachable nodes: {sorted(unreachable)}")
        if not any(END in self.successors(n) for n in self.nodes):
            raise GraphValidationError("no node transitions to END")

    def to_mermaid(self) -> str:
        """Render the topology (used in docs and, later, in the dashboard)."""
        lines = ["flowchart TD"]
        for name in self.nodes:
            for succ in sorted(self.successors(name)):
                target = "END" if succ == END else succ
                arrow = "-.->" if name in self._conditionals else "-->"
                lines.append(f"    {name} {arrow} {target}")
        return "\n".join(lines)
