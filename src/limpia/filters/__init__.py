"""Document quality filters."""

from __future__ import annotations

from limpia.config import LanguageConfig
from limpia.filters.fineweb import FineWebLineFilter
from limpia.filters.gopher import GopherQualityFilter, GopherRepetitionFilter
from limpia.pipeline import Filter

__all__ = [
    "FineWebLineFilter",
    "GopherQualityFilter",
    "GopherRepetitionFilter",
    "quality_filters",
]


def quality_filters(config: LanguageConfig) -> list[Filter]:
    """The default filter stack, cheapest and most selective first.

    Gopher quality rules are O(n) and remove most junk, so they run before the
    n-gram repetition statistics, which are the most expensive.

    With the bundled thresholds, Gopher's duplicate-line character rule cannot
    fire in this order: it divides the same duplicated-line characters by
    ``len(text)``, which is at least FineWeb's denominator, and its threshold
    (``> 0.20``) is above FineWeb's (``>= 0.10``), so FineWeb removes every such
    document first. The rule is kept for callers that run
    ``GopherRepetitionFilter`` on its own.
    """
    return [
        GopherQualityFilter(config.gopher_quality, config.stopwords),
        FineWebLineFilter(config.fineweb),
        GopherRepetitionFilter(config.gopher_repetition),
    ]
