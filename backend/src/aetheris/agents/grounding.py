"""Mechanical quote grounding.

LLM verifiers can "quote" text that is not in the source. Every quote is therefore
located in the evidence by code, in three increasingly tolerant passes:

1. ``exact``      – verbatim substring;
2. ``normalized`` – after case folding and collapsing whitespace / typographic
                    quotes and dashes (offsets are mapped back to the original);
3. ``fuzzy``      – difflib alignment covering >= ``min_coverage`` of the quote
                    within a compact window (tolerates a dropped comma or a
                    re-hyphenated word, not a paraphrase).

Quotes that fail all three are rejected.
"""

from __future__ import annotations

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


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalised text plus, for each normalised char, its index in the original."""
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for i, ch in enumerate(text.translate(_TRANSLATE).lower()):
        if ch.isspace():
            if previous_space:
                continue
            ch = " "
            previous_space = True
        else:
            previous_space = False
        chars.append(ch)
        index_map.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


def locate_quote(
    quote: str, text: str, *, min_coverage: float = 0.9, min_length: int = 12
) -> tuple[int, int, MatchKind] | None:
    """Return ``(start, end, kind)`` of ``quote`` inside ``text`` or ``None``."""
    quote = quote.strip().strip('"').strip()
    if len(quote) < min_length:
        return None

    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote), "exact"

    norm_text, index_map = _normalize_with_map(text)
    norm_quote, _ = _normalize_with_map(quote)
    if not norm_quote:
        return None
    pos = norm_text.find(norm_quote)
    if pos >= 0:
        return index_map[pos], index_map[pos + len(norm_quote) - 1] + 1, "normalized"

    matcher = SequenceMatcher(None, norm_quote, norm_text, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    if not blocks:
        return None
    covered = sum(b.size for b in blocks)
    first, last = blocks[0], blocks[-1]
    span_len = last.b + last.size - first.b
    if covered / len(norm_quote) < min_coverage or span_len > 1.25 * len(norm_quote):
        return None
    return index_map[first.b], index_map[last.b + last.size - 1] + 1, "fuzzy"
