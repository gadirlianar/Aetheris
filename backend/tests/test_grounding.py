"""Unit tests for 3-pass mechanical quote grounding and noisy PDF/LaTeX artifact alignment."""

from __future__ import annotations

from aetheris.agents.grounding import locate_quote
from aetheris.core.models import Paper
from aetheris.retrieval.text import chunk_paper, normalize_for_indexing

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


def test_noisy_pdf_line_wrap_hyphenation_and_latex_citation_grounding() -> None:
    noisy_abstract = (
        "In multi-column PDF extractions, retriev-\n"
        "al-augmented models [12, 14] achieve \\textbf{significant} gains "
        "of $10,000\\times$ over closed-book baselines \\cite{lewis2020rag} in accuracy."
    )
    # Clean LLM quote without line-wrap hyphen, inline citations, or LaTeX wrappers
    clean_quote = (
        "retrieval-augmented models achieve significant gains of 10,000x over closed-book baselines"
    )
    loc = locate_quote(clean_quote, noisy_abstract)
    assert loc is not None
    start, end, kind = loc
    assert kind in ("normalized", "fuzzy")
    sliced = noisy_abstract[start:end]
    assert sliced.startswith("retriev-\nal-augmented")
    assert sliced.endswith("closed-book baselines")


def test_chunk_paper_preserves_exact_source_offsets_on_noisy_text() -> None:
    raw = (
        "First sentence with PDF hyphen-\nation and inline citation [1, 2]. "
        "Second sentence with $O(N^2)$ math and \\emph{formatting} markers."
    )
    paper = Paper(
        id="arxiv:test",
        source="arxiv",
        title="Noisy PDF Test",
        abstract=raw,
        url="https://arxiv.org/abs/test",
    )
    chunks = chunk_paper(paper, sentences_per_chunk=1, stride=1)
    assert len(chunks) == 2
    for ch in chunks:
        assert raw[ch.char_start : ch.char_end] == ch.text

    cleaned = normalize_for_indexing(raw)
    assert "hyphenation" in cleaned
    assert "\\emph" not in cleaned
