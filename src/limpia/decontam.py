"""Benchmark decontamination by n-gram overlap.

Training on evaluation data inflates benchmark scores without improving the
model. The standard check, used for GPT-3 with 13-grams, is to look for long
word n-grams shared between a training document and any benchmark example: a
13-gram match is very unlikely to happen by chance in natural text, so it is a
high-precision signal of leakage.

Both sides are normalized identically (NFKC, casefold, punctuation removed) so
that a leaked passage is still found after reformatting. Only hashes of the
benchmark n-grams are stored, which keeps the index small and means the
benchmark text itself does not need to be shipped alongside the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from limpia.document import Document
from limpia.hashing import Hasher, get_hash_fn
from limpia.pipeline import Filter, Transform
from limpia.text import normalize_for_matching


def _ngram_hashes(text: str, n: int, hash_fn) -> list[int]:
    words = normalize_for_matching(text).split()
    return [hash_fn(" ".join(words[i : i + n]).encode("utf-8")) for i in range(len(words) - n + 1)]


class NGramIndex:
    """Set of hashed word n-grams from benchmark texts.

    Benchmark texts shorter than ``n`` words cannot produce an n-gram; they are
    counted in ``skipped_short`` rather than silently ignored, so the caller
    can decide to lower ``n`` for benchmarks with short items.
    """

    def __init__(self, n: int = 13, hasher: Hasher = "blake2b") -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self._hash = get_hash_fn(hasher)
        self._grams: set[int] = set()
        self.indexed_texts = 0
        self.skipped_short = 0

    @classmethod
    def from_texts(cls, texts: Iterable[str], n: int = 13, hasher: Hasher = "blake2b"):
        index = cls(n, hasher)
        for text in texts:
            index.add(text)
        return index

    def add(self, text: str) -> None:
        grams = _ngram_hashes(text, self.n, self._hash)
        if not grams:
            self.skipped_short += 1
            return
        self._grams.update(grams)
        self.indexed_texts += 1

    def __len__(self) -> int:
        return len(self._grams)

    def overlap(self, text: str) -> tuple[int, int]:
        """Return (matched n-grams, total n-grams) of ``text``."""
        grams = _ngram_hashes(text, self.n, self._hash)
        return sum(g in self._grams for g in grams), len(grams)


@dataclass(frozen=True)
class ContaminationPolicy:
    """A document is contaminated if it has at least ``min_matches`` matching
    n-grams and the matching fraction reaches ``min_fraction``.

    The default (any single match) is the conservative GPT-3 rule; raising
    ``min_fraction`` tolerates documents that merely quote a benchmark sentence
    inside a long, otherwise clean text.
    """

    min_matches: int = 1
    min_fraction: float = 0.0

    def is_contaminated(self, matched: int, total: int) -> bool:
        if matched < self.min_matches or total == 0:
            return False
        return matched / total >= self.min_fraction


class DecontaminationFilter(Filter):
    """Drop contaminated documents."""

    name = "decontam"

    def __init__(self, index: NGramIndex, policy: ContaminationPolicy | None = None) -> None:
        self.index = index
        self.policy = policy or ContaminationPolicy()

    def check(self, doc: Document) -> str | None:
        matched, total = self.index.overlap(doc.text)
        return "decontam:contaminated" if self.policy.is_contaminated(matched, total) else None


class DecontaminationTagger(Transform):
    """Keep every document but annotate overlap in ``metadata['contamination']``.

    Flagging instead of dropping is useful to measure contamination before
    deciding on a policy, or to remove it downstream per benchmark.
    """

    name = "decontam_flag"

    def __init__(self, index: NGramIndex, policy: ContaminationPolicy | None = None) -> None:
        self.index = index
        self.policy = policy or ContaminationPolicy()

    def apply(self, doc: Document) -> tuple[Document, Mapping[str, int]]:
        matched, total = self.index.overlap(doc.text)
        flagged = self.policy.is_contaminated(matched, total)
        new = doc.with_metadata(
            contamination={"matched_ngrams": matched, "total_ngrams": total, "flagged": flagged}
        )
        return new, {"flagged": int(flagged)}


def decontamination_step(
    index: NGramIndex,
    mode: Literal["drop", "flag"] = "drop",
    policy: ContaminationPolicy | None = None,
) -> Filter | Transform:
    if mode == "drop":
        return DecontaminationFilter(index, policy)
    if mode == "flag":
        return DecontaminationTagger(index, policy)
    raise ValueError(f"mode must be 'drop' or 'flag', got {mode!r}")
