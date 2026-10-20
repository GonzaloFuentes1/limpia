"""Text primitives shared by filters, deduplication and decontamination.

Every component tokenizes the same way so that a statistic computed by one step
(the word count of a quality filter) is comparable with one computed by another
(the n-grams used for decontamination). Tokenization is deliberately simple:
whitespace splitting plus Unicode punctuation stripping. Spanish and Portuguese
separate words with spaces, so this is adequate for them, and it avoids pinning
results to a trained tokenizer whose vocabulary would change the statistics.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r\f\v]*\n")


def _is_punct_or_symbol(ch: str) -> bool:
    # Unicode categories P* (punctuation) and S* (symbols) cover the Spanish
    # inverted marks, guillemets and dashes without maintaining a manual list.
    return unicodedata.category(ch)[0] in "PS"


def strip_punctuation(token: str) -> str:
    """Remove leading and trailing punctuation/symbols, keep inner ones.

    Inner characters are kept so that tokens such as ``e-mail`` or ``3,5`` stay
    single words, which matches how a reader would count them.
    """
    start, end = 0, len(token)
    while start < end and _is_punct_or_symbol(token[start]):
        start += 1
    while end > start and _is_punct_or_symbol(token[end - 1]):
        end -= 1
    return token[start:end]


def split_tokens(text: str) -> list[str]:
    """Whitespace tokens, punctuation included (used for alphabetic ratios)."""
    return text.split()


def split_words(text: str, *, lowercase: bool = False) -> list[str]:
    """Whitespace tokens with surrounding punctuation removed; empty tokens dropped."""
    words = []
    for token in text.split():
        word = strip_punctuation(token)
        if word:
            words.append(word.lower() if lowercase else word)
    return words


def split_lines(text: str) -> list[str]:
    """Non-empty lines with surrounding whitespace removed.

    Blank lines carry layout, not content; counting them would let a document
    pass a line-ratio filter just by adding empty lines.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs separated by one or more blank lines, stripped and non-empty."""
    return [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]


def normalize_for_matching(text: str) -> str:
    """Canonical form for duplicate and contamination matching.

    NFKC folds compatibility characters (full-width digits, ligatures) that
    differ between scrapes of the same page; casefolding and punctuation
    removal make matching robust to trivial formatting changes. Accents are
    kept on purpose: in Spanish they are often lexical (``esta`` / ``está``).
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    chars = [" " if _is_punct_or_symbol(ch) else ch for ch in text]
    return _WHITESPACE.sub(" ", "".join(chars)).strip()


def word_ngrams(words: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """All contiguous n-grams of ``words`` (empty when ``len(words) < n``)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
