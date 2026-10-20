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
    """
    return [
        GopherQualityFilter(config.gopher_quality, config.stopwords),
        FineWebLineFilter(config.fineweb),
        GopherRepetitionFilter(config.gopher_repetition),
    ]
