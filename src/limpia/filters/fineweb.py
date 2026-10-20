"""FineWeb line-level heuristics (Penedo et al., 2024, arXiv:2406.17557).

FineWeb added three rules on top of Gopher and C4 that target the line
structure of extracted web pages: lists without sentence punctuation, pages
dominated by short navigation lines, and boilerplate lines repeated within a
page. The comparisons follow the paper's direction exactly (``<=`` for the
punctuation ratio, ``>=`` for the other two), which differs from Gopher's
strict ``>``; keeping them as published makes results comparable.
"""

from __future__ import annotations

from limpia.config import FineWebConfig
from limpia.document import Document
from limpia.filters.repetition import duplicate_fractions
from limpia.pipeline import Filter
from limpia.text import split_lines


class FineWebLineFilter(Filter):
    name = "fineweb_lines"

    def __init__(self, config: FineWebConfig) -> None:
        self.config = config

    def measure(self, text: str) -> dict[str, float]:
        lines = split_lines(text)
        if not lines:
            return {"terminal_punct_ratio": 0.0, "short_line_ratio": 0.0, "dup_line_char_frac": 0.0}
        terminal = tuple(self.config.terminal_punctuation)
        _, dup_chars = duplicate_fractions(lines)
        return {
            "terminal_punct_ratio": sum(line.endswith(terminal) for line in lines) / len(lines),
            "short_line_ratio": (
                sum(len(line) < self.config.short_line_chars for line in lines) / len(lines)
            ),
            "dup_line_char_frac": dup_chars,
        }

    def check(self, doc: Document) -> str | None:
        c = self.config
        if not split_lines(doc.text):
            return "fineweb_lines:empty"
        m = self.measure(doc.text)
        if m["terminal_punct_ratio"] <= c.min_terminal_punct_ratio:
            return "fineweb_lines:terminal_punct_ratio"
        if m["short_line_ratio"] >= c.max_short_line_ratio:
            return "fineweb_lines:short_line_ratio"
        if m["dup_line_char_frac"] >= c.max_dup_line_char_frac:
            return "fineweb_lines:dup_line_char_frac"
        return None
