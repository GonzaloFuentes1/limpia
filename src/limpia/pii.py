"""PII scrubbing for Chilean and general web text.

Identifiers are replaced by typed tags (``<EMAIL>``, ``<RUT>``, ``<PHONE>``,
``<IP>``) rather than deleted, so sentences stay grammatical and a model still
learns that an identifier appears in that position, without memorizing it.

Precision matters as much as recall here: replacing arbitrary numbers would
corrupt prices, dates and statistics. The RUT matcher therefore validates the
mod-11 check digit and leaves invalid candidates untouched, and the phone
matcher requires the ``+56`` country code.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from limpia.document import Document
from limpia.pipeline import Transform

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}(?![\w-])"
)

# Dotted (12.345.678-5) or plain (12345678-5) body, then the check digit.
RUT_RE = re.compile(r"(?<![\w.])(\d{1,2}\.\d{3}\.\d{3}|\d{7,8})-([\dkK])(?![\w-])")

# +56 followed by exactly nine digits, with optional spaces, dots, dashes or
# parentheses between them: +56 9 1234 5678, +56 (2) 2345 6789, +56-32-123-4567.
PHONE_RE = re.compile(r"(?<![\w+])(?:\(\+56\)|\+56)(?:[\s.\-()]*\d){9}(?![\d])")

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.])(?:{_OCTET}\.){{3}}{_OCTET}(?!\.?\d)")


def rut_check_digit(body: int) -> str:
    """Chilean RUT check digit (modulo 11 with weights 2..7 from the right)."""
    total, weight = 0, 2
    while body > 0:
        total += (body % 10) * weight
        body //= 10
        weight = 2 if weight == 7 else weight + 1
    remainder = 11 - total % 11
    return {11: "0", 10: "K"}.get(remainder, str(remainder))


def is_valid_rut(text: str) -> bool:
    """Whether ``text`` (e.g. ``12.345.678-5``) has a correct check digit."""
    m = RUT_RE.fullmatch(text.strip())
    if not m:
        return False
    body = int(m.group(1).replace(".", ""))
    return rut_check_digit(body) == m.group(2).upper()


def _is_chilean_phone(match: str) -> bool:
    digits = re.sub(r"\D", "", match)[2:]  # drop the 56 country code
    # Chilean numbers have nine digits; the first is never 0 or 1.
    return len(digits) == 9 and digits[0] in "23456789"


@dataclass(frozen=True)
class _Rule:
    tag: str
    pattern: re.Pattern[str]
    accept: Callable[[re.Match[str]], bool]


_RULES: tuple[_Rule, ...] = (
    _Rule("EMAIL", EMAIL_RE, lambda m: True),
    _Rule("RUT", RUT_RE, lambda m: is_valid_rut(m.group(0))),
    _Rule("PHONE", PHONE_RE, lambda m: _is_chilean_phone(m.group(0))),
    _Rule("IP", IPV4_RE, lambda m: True),
)


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII with typed tags; return the new text and counts per tag.

    Rules run in a fixed order (email first, so that digits inside an address
    are not mistaken for a phone or RUT), each on the output of the previous.
    """
    counts: dict[str, int] = {}
    for rule in _RULES:
        n = 0

        def _sub(m: re.Match[str], rule: _Rule = rule) -> str:
            nonlocal n
            if rule.accept(m):
                n += 1
                return f"<{rule.tag}>"
            return m.group(0)

        text = rule.pattern.sub(_sub, text)
        counts[rule.tag] = n
    return text, counts


class PIIScrubber(Transform):
    name = "pii"

    def apply(self, doc: Document) -> tuple[Document, Mapping[str, int]]:
        text, counts = scrub(doc.text)
        return (doc.with_text(text) if text != doc.text else doc), counts
