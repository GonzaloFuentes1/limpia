"""Repetition statistics shared by the Gopher and FineWeb filters.

These are pure functions of a list of strings so that each one can be tested
against an independent reference, separately from the threshold logic that
consumes them.

Every character fraction takes its denominator explicitly. The filters pass
the character length of the whole document (``len(text)``, whitespace
included), as Gopher (Rae et al., 2021, Appendix A) describes and as
datatrove's ``GopherRepetitionFilter`` implements, so that the published
thresholds keep their meaning. Dividing by word characters only would inflate
every fraction by the share of whitespace and punctuation in the text (about
20% for ordinary Spanish prose) and make the same thresholds stricter.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def duplicate_fractions(items: Sequence[str], *, total_chars: int) -> tuple[float, float]:
    """Return (fraction of items that repeat an earlier item, their character fraction).

    The first occurrence of an item is not counted as a duplicate: a page that
    contains a sentence once is fine, the second copy is the redundancy. The
    character fraction is the characters of the repeated items divided by
    ``total_chars``.
    """
    if not items:
        return 0.0, 0.0
    seen: set[str] = set()
    dup_items = 0
    dup_chars = 0
    for item in items:
        if item in seen:
            dup_items += 1
            dup_chars += len(item)
        else:
            seen.add(item)
    return dup_items / len(items), (dup_chars / total_chars if total_chars > 0 else 0.0)


def top_ngram_char_fraction(words: Sequence[str], n: int, *, total_chars: int) -> float:
    """Fraction of characters contained within the most frequent word n-gram.

    This is Gopher's statistic: pick the single most frequent n-gram, then
    divide ``count * len(ngram)`` by ``total_chars``, where ``len(ngram)`` is the
    length of the n-gram's words joined by single spaces. Ties between equally
    frequent n-grams go to the one that occurs first, as in datatrove
    (``Counter.most_common``). An n-gram that occurs only once is not
    repetition, so a document without repeated n-grams scores 0 (datatrove
    scores ``len(ngram) / len(text)`` there, which is far below any threshold
    for documents that pass the minimum word count). Overlapping occurrences
    (``"a a a"``) are counted separately, as in the reference implementation,
    so the value is clipped at 1.
    """
    if total_chars <= 0 or len(words) < n:
        return 0.0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = counts.most_common(1)[0]
    if count < 2:
        return 0.0
    return min(count * len(" ".join(gram)) / total_chars, 1.0)


def duplicated_ngram_char_fraction(words: Sequence[str], n: int, *, total_chars: int) -> float:
    """Fraction of characters inside word n-grams that already occurred earlier.

    A word is marked if it belongs to any n-gram occurrence whose n-gram has
    appeared at an earlier position; the numerator is the characters of the
    marked words. Overlapping occurrences mark each word at most once, which is
    Gopher's instruction not to count characters in overlapping n-grams more
    than once, so the result stays in [0, 1] even for text such as
    ``"a a a a a"``.
    """
    if total_chars <= 0 or len(words) < n:
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
    covered = sum(len(w) for w, m in zip(words, marked, strict=True) if m)
    return min(covered / total_chars, 1.0)
