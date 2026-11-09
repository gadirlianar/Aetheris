"""Unit tests for 3-pass mechanical quote grounding."""

from __future__ import annotations

from aetheris.agents.grounding import locate_quote

ABSTRACT = (
    "In controlled distraction experiments across 4,200 scientific queries, "
    "injecting topically similar—but factually irrelevant—passages into the prompt "
    "increases unfaithful synthesis claims from 8.1% to 27.4%."
)


def test_exact_quote_grounding() -> None:
    quote = "increases unfaithful synthesis claims from 8.1% to 27.4%."
    loc = locate_quote(quote, ABSTRACT)
    assert loc is not None
    start, end, kind = loc
    assert kind == "exact"
    assert ABSTRACT[start:end] == quote


def test_normalized_quote_grounding_handles_dashes_and_whitespace() -> None:
    quote = "injecting topically similar - but factually irrelevant - passages   into the prompt"
    loc = locate_quote(quote, ABSTRACT)
    assert loc is not None
    start, end, kind = loc
    assert kind in ("normalized", "fuzzy")
    assert "injecting topically similar" in ABSTRACT[start:end]


def test_fuzzy_quote_grounding_and_rejection_of_paraphrase() -> None:
    # Minor dropped comma still aligns >= 90%
    near = (
        "In controlled distraction experiments across 4200 scientific queries "
        "injecting topically similar"
    )
    loc = locate_quote(near, ABSTRACT)
    assert loc is not None
    assert loc[2] == "fuzzy"

    # Paraphrase not in source must be rejected
    paraphrase = (
        "Adding distracting documents to language model prompts triples the hallucination rate."
    )
    assert locate_quote(paraphrase, ABSTRACT) is None
    assert locate_quote("too short", ABSTRACT) is None
