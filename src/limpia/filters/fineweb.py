"""FineWeb line-level heuristics (Penedo et al., 2024, arXiv:2406.17557).

FineWeb added three rules on top of Gopher and C4 that target the line
structure of extracted web pages: lists without sentence punctuation, pages
dominated by short navigation lines, and boilerplate lines repeated within a
page. The comparisons follow the wording of the paper's section 3.6 (``<=``
for the punctuation ratio, ``>=`` for the other two, "shorter than 30
characters"), which differs from Gopher's strict ``>``.

datatrove's ``FineWebQualityFilter``, the code that produced FineWeb, differs
from that wording: it removes when the punctuation ratio is ``< 0.12``, the
short-line ratio is ``> 0.67`` with short meaning ``len(line) <= 30``, and the
duplicated-line character ratio is ``> 0.01`` (not 0.1). It also has a fourth
``list_ratio`` rule (newlines per word ``> 0.3``) that the paper's section 3.6
does not list and that is not implemented here. Results are therefore close
to, but not identical with, FineWeb's released filtering. The denominator of
the duplicated-line character ratio follows datatrove (document characters
without newlines), since the paper does not state one.
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
        # Denominator as in datatrove's FineWebQualityFilter: document characters
        # without newlines.
        _, dup_chars = duplicate_fractions(lines, total_chars=len(text.replace("\n", "")))
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
