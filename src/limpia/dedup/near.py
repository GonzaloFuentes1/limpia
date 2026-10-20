"""Near-duplicate removal: MinHash signatures, LSH banding and union-find."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from limpia.dedup.lsh import lsh_clusters, optimal_bands_rows
from limpia.dedup.minhash import MinHasher, Unit
from limpia.document import Document
from limpia.hashing import Hasher


@dataclass(frozen=True)
class MinHashDedupConfig:
    """Near-dedup parameters.

    Word 5-gram shingles follow common practice for web text (FineWeb and
    text-dedup both shingle on word 5-grams). ``bands``/``rows`` are derived
    from the threshold unless given explicitly.
    """

    threshold: float = 0.8
    num_perm: int = 128
    ngram: int = 5
    unit: Unit = "word"
    seed: int = 0
    hasher: Hasher = "blake2b"
    bands: int | None = None
    rows: int | None = None
    verify: bool = False


class MinHashDeduplicator:
    def __init__(self, config: MinHashDedupConfig | None = None) -> None:
        self.config = config or MinHashDedupConfig()
        c = self.config
        if (c.bands is None) != (c.rows is None):
            raise ValueError("set both bands and rows, or neither")
        if c.bands is None or c.rows is None:
            self.bands, self.rows = optimal_bands_rows(c.threshold, c.num_perm)
        else:
            self.bands, self.rows = c.bands, c.rows
        self.minhasher = MinHasher(
            c.num_perm, seed=c.seed, ngram=c.ngram, unit=c.unit, hasher=c.hasher
        )

    def signatures(self, texts: Iterable[str]) -> np.ndarray:
        return self.minhasher.signatures(texts)

    def cluster(self, signatures: np.ndarray) -> np.ndarray:
        """Root index per document; a document is kept iff it is its own root."""
        return lsh_clusters(
            signatures,
            self.bands,
            self.rows,
            verify_threshold=self.config.threshold if self.config.verify else None,
        )

    def deduplicate(self, docs: Sequence[Document]) -> tuple[list[Document], np.ndarray]:
        """In-memory convenience: return kept documents and the cluster roots."""
        roots = self.cluster(self.signatures(d.text for d in docs))
        kept = [d for i, d in enumerate(docs) if roots[i] == i]
        return kept, roots
