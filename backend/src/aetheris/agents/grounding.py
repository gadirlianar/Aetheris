"""Mechanical quote grounding with noisy PDF, LaTeX, and citation artifact tolerance.

LLM verifiers can "quote" text that is not in the source, or quote clean prose from
a multi-column PDF/LaTeX abstract that contains line-wrap hyphenations, inline math
delimiters, or bracketed citation markers. Every quote is therefore located in the
evidence by code, in three increasingly tolerant passes while strictly preserving
exact ``(char_start, char_end)`` offsets into the raw source string:

1. ``exact``      – verbatim substring;
2. ``normalized`` – after case folding, collapsing whitespace/typographic quotes/dashes,
                    rejoining line-wrap PDF hyphenations (``retriev-\\nal`` -> ``retrieval``),
                    stripping LaTeX math/formatting wrappers (``$p < 0.01$``, ``\\textbf{...}``),
                    and optionally eliding inline citation markers (``[12, 14]``, ``\\cite{...}``)
                    with an exact character-by-character inverse index map to the original text;
3. ``fuzzy``      – difflib alignment covering >= ``min_coverage`` of the quote
                    within a compact window (tolerates a dropped comma or minor OCR noise,
                    never a fabricated paraphrase).

Quotes that fail all three are rejected.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from aetheris.core.models import MatchKind

_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)

# Line-wrap hyphenation: a letter followed by '-' (or soft hyphen), optional spaces,
# a newline, optional spaces, and another lowercase letter.
_LINEWRAP_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-\s*\r?\n\s*(?=[a-z])")

# Inline citation markers such as [12], [3, 4], [1-5], or \cite{smith2023}, \citep{...}
_INLINE_CITATION_RE = re.compile(
    r"\\cite[tp]?\s*(?:\[[^\]]*\])?\{[^}]*\}|\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]"
)

# Formatting wrappers where the command name should be skipped but the inner text kept
_LATEX_WRAPPER_RE = re.compile(r"\\(?:textbf|textit|textrm|emph|mathrm|mathbf|mathit|text)\s*\{")

# Known LaTeX math symbol macros mapped to normalized ASCII equivalents
_LATEX_SYMBOLS: tuple[tuple[str, str], ...] = (
    (r"\times", "x"),
    (r"\pm", "+-"),
    (r"\mp", "-+"),
    (r"\leq", "<="),
    (r"\le", "<="),
    (r"\geq", ">="),
    (r"\ge", ">="),
    (r"\neq", "!="),
    (r"\approx", "~"),
    (r"\sim", "~"),
    (r"\infty", "inf"),
    (r"\%", "%"),
)


def _compute_skip_mask(text: str, *, strip_citations: bool) -> list[bool]:
    """Return a boolean mask of characters in ``text`` that represent structural noise."""
    n = len(text)
    skip = [False] * n

    for match in _LINEWRAP_HYPHEN_RE.finditer(text):
        for idx in range(match.start(), match.end()):
            skip[idx] = True

    for match in _LATEX_WRAPPER_RE.finditer(text):
        for idx in range(match.start(), match.end()):
            skip[idx] = True

    if strip_citations:
        for match in _INLINE_CITATION_RE.finditer(text):
            for idx in range(match.start(), match.end()):
                skip[idx] = True

    return skip


def _normalize_with_map(text: str, *, strip_citations: bool = False) -> tuple[str, list[int]]:
    """Normalized text plus, for each normalized char, its exact index in ``text``."""
    n = len(text)
    skip = _compute_skip_mask(text, strip_citations=strip_citations)
    translated = text.translate(_TRANSLATE).lower()

    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True

    def emit(ch: str, orig_idx: int) -> None:
        nonlocal previous_space
        if ch.isspace():
            if previous_space:
                return
            chars.append(" ")
            index_map.append(orig_idx)
            previous_space = True
        else:
            chars.append(ch)
            index_map.append(orig_idx)
            previous_space = False

    i = 0
    while i < n:
        if skip[i]:
            i += 1
            continue
        raw_ch = text[i]
        # Skip soft hyphens, LaTeX math mode delimiters '$', and bare braces '{' '}'
        if raw_ch in ("\u00ad", "$", "{", "}"):
            i += 1
            continue
        # Handle LaTeX macros starting with backslash
        if raw_ch == "\\":
            matched_sym = False
            for macro, replacement in _LATEX_SYMBOLS:
                if text.startswith(macro, i):
                    for r_ch in replacement:
                        emit(r_ch, i)
                    i += len(macro)
                    matched_sym = True
                    break
            if matched_sym:
                continue
            # For Greek letters or spacing macros (\alpha, \beta, \, \;), drop the '\'
            if i + 1 < n and text[i + 1] in (",", ";", ":", "!"):
                emit(" ", i)
                i += 2
                continue
            i += 1
            continue

        emit(translated[i], i)
        i += 1

    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


def locate_quote(
    quote: str, text: str, *, min_coverage: float = 0.9, min_length: int = 12
) -> tuple[int, int, MatchKind] | None:
    """Return ``(start, end, kind)`` of ``quote`` inside ``text`` or ``None``.

    ``start`` and ``end`` are guaranteed to be valid slice indices into the
    original, unmodified ``text`` string.
    """
    quote = quote.strip().strip('"').strip()
    if len(quote) < min_length:
        return None

    # Pass 1: Exact substring match
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote), "exact"

    # Pass 2a: Normalized match (preserving citations)
    norm_text, index_map = _normalize_with_map(text, strip_citations=False)
    norm_quote, _ = _normalize_with_map(quote, strip_citations=False)
    if not norm_quote:
        return None
    pos = norm_text.find(norm_quote)
    if pos >= 0:
        return index_map[pos], index_map[pos + len(norm_quote) - 1] + 1, "normalized"

    # Pass 2b: Normalized match with inline citation markers elided
    norm_text_nocite, index_map_nocite = _normalize_with_map(text, strip_citations=True)
    norm_quote_nocite, _ = _normalize_with_map(quote, strip_citations=True)
    if norm_quote_nocite:
        pos_nc = norm_text_nocite.find(norm_quote_nocite)
        if pos_nc >= 0:
            return (
                index_map_nocite[pos_nc],
                index_map_nocite[pos_nc + len(norm_quote_nocite) - 1] + 1,
                "normalized",
            )

    # Pass 3: Fuzzy contiguous-window SequenceMatcher alignment
    target_norm = norm_text_nocite if norm_quote_nocite else norm_text
    target_map = index_map_nocite if norm_quote_nocite else index_map
    query_norm = norm_quote_nocite or norm_quote

    matcher = SequenceMatcher(None, query_norm, target_norm, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    if not blocks:
        return None
    covered = sum(b.size for b in blocks)
    first, last = blocks[0], blocks[-1]
    span_len = last.b + last.size - first.b
    if covered / len(query_norm) < min_coverage or span_len > 1.25 * len(query_norm):
        return None
    return target_map[first.b], target_map[last.b + last.size - 1] + 1, "fuzzy"
