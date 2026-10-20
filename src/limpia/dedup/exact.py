"""Exact deduplication on normalized text.

Exact duplicates are the cheapest redundancy to remove and the most common in
crawls (mirrors, syndicated news, boilerplate pages). Matching on a normalized
form catches copies that differ only in case, punctuation or whitespace,
which a raw byte hash would miss.

A 64-bit hash is stored instead of the text to keep memory proportional to the
number of distinct documents, not their size. The birthday bound gives a
collision probability of roughly ``N**2 / 2**65``: about 3e-4 for 100 million
documents, which would wrongly drop a single document.

A document whose normalized text is empty (empty, or only punctuation,
symbols or emoji) is never treated as a duplicate: normalization erased
everything that could distinguish two such documents, so matching them would
remove unrelated content. The same rule applies in MinHash dedup.
"""

from __future__ import annotations

import numpy as np

from limpia.document import Document
from limpia.hashing import Hasher, hash64
from limpia.pipeline import Filter
from limpia.text import normalize_for_matching


def exact_key(text: str, hasher: Hasher = "blake2b") -> int:
    return hash64(normalize_for_matching(text), hasher)


class ExactDedupFilter(Filter):
    """Streaming filter that keeps the first occurrence of each normalized text."""

    name = "exact_dedup"

    def __init__(self, hasher: Hasher = "blake2b") -> None:
        self.hasher: Hasher = hasher
        self._seen: set[int] = set()

    def check(self, doc: Document) -> str | None:
        norm = normalize_for_matching(doc.text)
        if not norm:
            return None
        key = hash64(norm, self.hasher)
        if key in self._seen:
            return "exact_dedup:duplicate"
        self._seen.add(key)
        return None


def exact_clusters(texts: list[str], hasher: Hasher = "blake2b") -> np.ndarray:
    """Cluster root (index of first occurrence) for every text.

    Texts that normalize to the empty string are their own roots.
    """
    first: dict[int, int] = {}
    roots = np.empty(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        norm = normalize_for_matching(text)
        roots[i] = first.setdefault(hash64(norm, hasher), i) if norm else i
    return roots
