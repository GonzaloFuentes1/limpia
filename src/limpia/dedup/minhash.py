"""MinHash signatures (Broder, 1997).

For a random permutation ``pi`` of the shingle universe,
``P[min pi(A) == min pi(B)] = |A & B| / |A | B|``, so the fraction of equal
coordinates between two signatures is an unbiased estimate of the Jaccard
similarity, with binomial variance ``J (1 - J) / num_perm``.

Permutations are approximated by the universal family
``h(x) = (a x + b) mod p`` with ``p = 2**31 - 1``. The 31-bit prime is chosen so
that ``a * x + b < 2**63`` and the arithmetic is exact in ``uint64``: a 61-bit
prime (as used in some implementations) overflows 64-bit products and silently
turns the modular hash into a different, non-universal function. The cost is a
31-bit shingle space; two distinct shingles collide with probability about
``2**-31``, far below the estimator's own sampling error.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from limpia.hashing import Hasher, get_hash_fn
from limpia.text import normalize_for_matching

PRIME = (1 << 31) - 1
EMPTY = PRIME  # Larger than any hash value: the signature of an empty set.
_CHUNK = 4096

Unit = Literal["word", "char"]


def shingles(text: str, n: int = 5, unit: Unit = "word") -> set[str]:
    """Normalized word or character n-gram shingles.

    Texts shorter than ``n`` units yield a single shingle with the whole text,
    so that short documents can still be matched with each other instead of
    all mapping to the empty set.
    """
    norm = normalize_for_matching(text)
    if unit == "word":
        units: list[str] = norm.split()
        sep = " "
    elif unit == "char":
        units = list(norm)
        sep = ""
    else:
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")
    if not units:
        return set()
    if len(units) < n:
        return {sep.join(units)}
    return {sep.join(units[i : i + n]) for i in range(len(units) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity (1.0 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class MinHasher:
    """Compute MinHash signatures with ``num_perm`` hash functions.

    The seed fully determines the hash family, so signatures are reproducible
    across runs and machines (given the same ``hasher``).
    """

    def __init__(
        self,
        num_perm: int = 128,
        *,
        seed: int = 0,
        ngram: int = 5,
        unit: Unit = "word",
        hasher: Hasher = "blake2b",
    ) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.ngram = ngram
        self.unit: Unit = unit
        self._hash = get_hash_fn(hasher)
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, PRIME, size=num_perm, dtype=np.uint64)

    def hash_shingles(self, items: Iterable[str]) -> np.ndarray:
        """Map shingles to integers in ``[0, PRIME)``."""
        values = [self._hash(s.encode("utf-8")) % PRIME for s in items]
        return np.asarray(values, dtype=np.uint64)

    def signature_from_hashes(self, hashes: np.ndarray) -> np.ndarray:
        """Signature of a set given its shingle hashes (``uint32``, length ``num_perm``)."""
        sig = np.full(self.num_perm, EMPTY, dtype=np.uint64)
        hashes = np.asarray(hashes, dtype=np.uint64)
        # Chunking bounds the (num_perm x chunk) intermediate for very long documents.
        for start in range(0, hashes.size, _CHUNK):
            chunk = hashes[start : start + _CHUNK]
            vals = (np.multiply.outer(self.a, chunk) + self.b[:, None]) % np.uint64(PRIME)
            np.minimum(sig, vals.min(axis=1), out=sig)
        return sig.astype(np.uint32)

    def signature(self, text: str) -> np.ndarray:
        return self.signature_from_hashes(self.hash_shingles(shingles(text, self.ngram, self.unit)))

    def signatures(self, texts: Iterable[str]) -> np.ndarray:
        """Stack signatures into an ``(n_docs, num_perm)`` ``uint32`` matrix."""
        rows = [self.signature(t) for t in texts]
        if not rows:
            return np.empty((0, self.num_perm), dtype=np.uint32)
        return np.vstack(rows)
