"""Repetition statistics shared by the Gopher and FineWeb filters.

These are pure functions of a list of strings so that each one can be tested
against an independent brute-force reference, separately from the threshold
logic that consumes them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def duplicate_fractions(items: Sequence[str]) -> tuple[float, float]:
    """Return (fraction of items that repeat an earlier item, their char fraction).

    The first occurrence of an item is not counted as a duplicate: a page that
    contains a sentence once is fine, the second copy is the redundancy.
    """
    if not items:
        return 0.0, 0.0
    seen: set[str] = set()
    dup_items = 0
    dup_chars = 0
    total_chars = 0
    for item in items:
        total_chars += len(item)
        if item in seen:
            dup_items += 1
            dup_chars += len(item)
        else:
            seen.add(item)
    return dup_items / len(items), (dup_chars / total_chars if total_chars else 0.0)


def top_ngram_char_fraction(words: Sequence[str], n: int) -> float:
    """Fraction of word characters covered by the most frequent repeated n-gram.

    Characters are counted over words only (whitespace excluded) so that the
    statistic does not depend on spacing. An n-gram that occurs once is not
    repetition, so documents without any repeated n-gram score 0. Ties between
    equally frequent n-grams are broken towards the longer one, i.e. towards
    the more conservative (higher) score.
    """
    total = sum(len(w) for w in words)
    if total == 0 or len(words) < n:
        return 0.0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    best = 0
    for gram, count in counts.items():
        if count > 1:
            best = max(best, count * sum(len(w) for w in gram))
    return min(best / total, 1.0)


def duplicated_ngram_char_fraction(words: Sequence[str], n: int) -> float:
    """Fraction of word characters inside n-grams that already occurred earlier.

    A word is marked if it belongs to any n-gram occurrence whose n-gram has
    appeared at an earlier position. Overlapping occurrences mark each word at
    most once, so the result is a fraction in [0, 1] even for text such as
    ``"a a a a a"``.
    """
    total = sum(len(w) for w in words)
    if total == 0 or len(words) < n:
        return 0.0
    marked = [False] * len(words)
    seen: set[tuple[str, ...]] = set()
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        if gram in seen:
            for j in range(i, i + n):
                marked[j] = True
        else:
            seen.add(gram)
    return sum(len(w) for w, m in zip(words, marked, strict=True) if m) / total
