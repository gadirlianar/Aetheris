"""Tokenisation, PDF/LaTeX-aware text normalization, sentence segmentation and chunking."""

from __future__ import annotations

import re

from aetheris.core.models import Chunk, Paper

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_LINEWRAP_RE = re.compile(r"(?<=[A-Za-z])-\s*\r?\n\s*(?=[a-z])")
_LATEX_CMD_RE = re.compile(r"\\(?:textbf|textit|textrm|emph|mathrm|mathbf|text|cite[tp]?)\b")

STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "us",
        "using",
        "use",
        "used",
        "via",
    }
)

_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "et al",
        "al",
        "fig",
        "figs",
        "eq",
        "eqs",
        "vs",
        "cf",
        "approx",
        "ref",
        "refs",
        "sec",
        "no",
        "dr",
        "mr",
        "ms",
        "prof",
        "resp",
        "ca",
        "viz",
    }
)

_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]*\s+")


def normalize_for_indexing(text: str) -> str:
    """Repair PDF line-wrap hyphenations and strip LaTeX command wrappers for indexing.

    Note: This is used only when extracting lexical tokens for BM25/dense indexing;
    ``chunk_paper`` and ``split_sentences`` operate on the raw ``Paper.abstract`` so
    that ``char_start`` and ``char_end`` offsets remain 100% exact.
    """
    repaired = _LINEWRAP_RE.sub("", text.replace("\u00ad", ""))
    return _LATEX_CMD_RE.sub(" ", repaired)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    cleaned = normalize_for_indexing(text).lower()
    raw_tokens = _TOKEN_RE.findall(cleaned)
    tokens = [t.strip(".-") for t in raw_tokens if t.strip(".-")]
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of sentences in ``text``.

    Conservative: it refuses to split after common scientific abbreviations
    ("e.g.", "et al.", "Fig.") and before lowercase continuations.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end()
        nxt = text[end : end + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([“$\\"):
            continue
        head = text[start : match.start()].rstrip()
        last_word = head.rsplit(None, 1)[-1].lower() if head else ""
        tail = head[-6:].lower()
        if last_word in _ABBREVIATIONS or tail.endswith("et al") or re.search(r"\b[a-z]$", tail):
            continue
        spans.append(_trim(text, start, match.start() + 1))
        start = end
    if text[start:].strip():
        spans.append(_trim(text, start, len(text)))
    return [s for s in spans if s[1] > s[0]]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_paper(paper: Paper, *, sentences_per_chunk: int = 3, stride: int = 2) -> list[Chunk]:
    """Split an abstract into overlapping sentence windows with exact offsets."""
    text = paper.abstract
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[Chunk] = []
    index = 0
    position = 0
    while True:
        window = sentences[position : position + sentences_per_chunk]
        start, end = window[0][0], window[-1][1]
        chunks.append(
            Chunk(
                id=f"{paper.id}#{index}",
                paper_id=paper.id,
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
        index += 1
        if position + sentences_per_chunk >= len(sentences):
            break
        position += stride
    return chunks
