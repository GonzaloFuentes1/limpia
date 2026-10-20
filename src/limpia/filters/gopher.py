"""Gopher document heuristics (Rae et al., 2021, Appendix A, arXiv:2112.11446).

Each filter separates *measuring* (``measure`` returns the raw statistics) from
*deciding* (``check`` compares them with thresholds). The split makes the
statistics testable in isolation and lets callers log them to calibrate
thresholds on their own data before enforcing them.
"""

from __future__ import annotations

from limpia.config import GopherQualityConfig, GopherRepetitionConfig
from limpia.document import Document
from limpia.filters.repetition import (
    duplicate_fractions,
    duplicated_ngram_char_fraction,
    top_ngram_char_fraction,
)
from limpia.pipeline import Filter
from limpia.text import split_lines, split_paragraphs, split_tokens, split_words

_ELLIPSES = ("...", "…")


class GopherQualityFilter(Filter):
    """Word-count, word-length, symbol, bullet, ellipsis, alphabetic and stopword rules.

    The stopword rule is the one that most needs localization: Gopher's English
    list would reject every Spanish document, so the list comes from the
    language config.
    """

    name = "gopher_quality"

    def __init__(self, config: GopherQualityConfig, stopwords: frozenset[str]) -> None:
        self.config = config
        self.stopwords = stopwords

    def measure(self, text: str) -> dict[str, float]:
        words = split_words(text)
        tokens = split_tokens(text)
        lines = split_lines(text)
        n_words = len(words)
        bullets = tuple(self.config.bullet_prefixes)
        symbols = text.count("#") + sum(text.count(e) for e in _ELLIPSES)
        return {
            "n_words": float(n_words),
            "mean_word_length": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
            "symbol_word_ratio": (symbols / n_words) if n_words else 0.0,
            "bullet_lines_ratio": (
                sum(line.startswith(bullets) for line in lines) / len(lines) if lines else 0.0
            ),
            "ellipsis_lines_ratio": (
                sum(line.endswith(_ELLIPSES) for line in lines) / len(lines) if lines else 0.0
            ),
            "alpha_words_ratio": (
                sum(any(c.isalpha() for c in t) for t in tokens) / len(tokens) if tokens else 0.0
            ),
            "stopwords": float(sum(w.lower() in self.stopwords for w in words)),
        }

    def check(self, doc: Document) -> str | None:
        c = self.config
        m = self.measure(doc.text)
        if m["n_words"] < c.min_words:
            return "gopher_quality:too_few_words"
        if m["n_words"] > c.max_words:
            return "gopher_quality:too_many_words"
        if m["mean_word_length"] < c.min_mean_word_length:
            return "gopher_quality:short_mean_word_length"
        if m["mean_word_length"] > c.max_mean_word_length:
            return "gopher_quality:long_mean_word_length"
        if m["symbol_word_ratio"] > c.max_symbol_word_ratio:
            return "gopher_quality:symbol_word_ratio"
        if m["bullet_lines_ratio"] > c.max_bullet_lines_ratio:
            return "gopher_quality:bullet_lines"
        if m["ellipsis_lines_ratio"] > c.max_ellipsis_lines_ratio:
            return "gopher_quality:ellipsis_lines"
        if m["alpha_words_ratio"] < c.min_alpha_words_ratio:
            return "gopher_quality:alpha_words_ratio"
        if m["stopwords"] < c.min_stopwords:
            return "gopher_quality:too_few_stopwords"
        return None


class GopherRepetitionFilter(Filter):
    """Duplicate line/paragraph fractions and top / duplicated n-gram fractions.

    N-grams are computed over lowercased, punctuation-stripped words so that
    "Comprar ahora." and "comprar ahora" count as the same repeated phrase.
    """

    name = "gopher_repetition"

    def __init__(self, config: GopherRepetitionConfig) -> None:
        self.config = config

    def measure(self, text: str) -> dict[str, float]:
        lines = split_lines(text)
        paragraphs = split_paragraphs(text)
        words = split_words(text, lowercase=True)
        dup_line, dup_line_chars = duplicate_fractions(lines)
        dup_para, dup_para_chars = duplicate_fractions(paragraphs)
        stats = {
            "dup_line_frac": dup_line,
            "dup_line_char_frac": dup_line_chars,
            "dup_para_frac": dup_para,
            "dup_para_char_frac": dup_para_chars,
        }
        for n in self.config.max_top_ngram_char_frac:
            stats[f"top_{n}gram_char_frac"] = top_ngram_char_fraction(words, n)
        for n in self.config.max_dup_ngram_char_frac:
            stats[f"dup_{n}gram_char_frac"] = duplicated_ngram_char_fraction(words, n)
        return stats

    def check(self, doc: Document) -> str | None:
        c = self.config
        m = self.measure(doc.text)
        if m["dup_line_frac"] > c.max_dup_line_frac:
            return "gopher_repetition:dup_line_frac"
        if m["dup_para_frac"] > c.max_dup_para_frac:
            return "gopher_repetition:dup_para_frac"
        if m["dup_line_char_frac"] > c.max_dup_line_char_frac:
            return "gopher_repetition:dup_line_char_frac"
        if m["dup_para_char_frac"] > c.max_dup_para_char_frac:
            return "gopher_repetition:dup_para_char_frac"
        for n, limit in sorted(c.max_top_ngram_char_frac.items()):
            if m[f"top_{n}gram_char_frac"] > limit:
                return f"gopher_repetition:top_{n}gram"
        for n, limit in sorted(c.max_dup_ngram_char_frac.items()):
            if m[f"dup_{n}gram_char_frac"] > limit:
                return f"gopher_repetition:dup_{n}gram"
        return None
