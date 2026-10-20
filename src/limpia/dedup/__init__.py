"""Exact and near-duplicate detection."""

from limpia.dedup.exact import ExactDedupFilter, exact_clusters
from limpia.dedup.lsh import collision_probability, lsh_clusters, optimal_bands_rows
from limpia.dedup.metrics import pairwise_scores
from limpia.dedup.minhash import MinHasher, estimate_jaccard, jaccard, shingles
from limpia.dedup.near import MinHashDedupConfig, MinHashDeduplicator
from limpia.dedup.unionfind import UnionFind

__all__ = [
    "ExactDedupFilter",
    "MinHashDedupConfig",
    "MinHashDeduplicator",
    "MinHasher",
    "UnionFind",
    "collision_probability",
    "estimate_jaccard",
    "exact_clusters",
    "jaccard",
    "lsh_clusters",
    "optimal_bands_rows",
    "pairwise_scores",
    "shingles",
]
