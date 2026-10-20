"""Disjoint-set forest whose roots are always the smallest member index."""

from __future__ import annotations


class UnionFind:
    """Union-find with path halving.

    Union always attaches the larger root under the smaller one. That gives up
    union-by-rank, but makes the representative of every set its smallest
    index, so "keep the representative" means "keep the first occurrence"
    regardless of the order in which unions happen. Path halving alone keeps
    finds amortized logarithmic, which is ample for dedup workloads.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo
        return lo
