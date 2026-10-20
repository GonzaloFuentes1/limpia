"""LSH banding and the choice of (bands, rows).

Splitting a signature of ``b * r`` values into ``b`` bands of ``r`` rows, two
documents with Jaccard ``s`` share at least one identical band with
probability ``P(s) = 1 - (1 - s**r)**b``, an S-curve whose steep part sits near
``(1 / b) ** (1 / r)``. For a target threshold ``t`` we want ``P`` close to 0
below ``t`` (false positives) and close to 1 above it (false negatives).

Following the approach popularized by datasketch, ``optimal_bands_rows``
minimizes the weighted areas

    FP(b, r) = integral_0^t P(s) ds        FN(b, r) = integral_t^1 1 - P(s) ds

over all ``b * r <= num_perm``. The integrals are computed with composite
Simpson quadrature; the tests check them against exact rational integration of
the binomial expansion of ``P``.
"""

from __future__ import annotations

import numpy as np

from limpia.dedup.unionfind import UnionFind

_QUAD_POINTS = 2001  # odd, required by Simpson's rule


def collision_probability(s: float | np.ndarray, bands: int, rows: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return 1.0 - (1.0 - s**rows) ** bands


def _simpson(y: np.ndarray, a: float, b: float) -> float:
    h = (b - a) / (len(y) - 1)
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def false_positive_area(threshold: float, bands: int, rows: int) -> float:
    x = np.linspace(0.0, threshold, _QUAD_POINTS)
    return _simpson(collision_probability(x, bands, rows), 0.0, threshold)


def false_negative_area(threshold: float, bands: int, rows: int) -> float:
    x = np.linspace(threshold, 1.0, _QUAD_POINTS)
    return _simpson(1.0 - collision_probability(x, bands, rows), threshold, 1.0)


def optimal_bands_rows(
    threshold: float,
    num_perm: int,
    *,
    fp_weight: float = 0.5,
    fn_weight: float = 0.5,
) -> tuple[int, int]:
    """Return ``(bands, rows)`` minimizing the weighted FP + FN area.

    Ties are broken towards fewer bands (iteration order), which uses fewer
    hash tables for the same error.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        for rows in range(1, num_perm // bands + 1):
            err = fp_weight * false_positive_area(
                threshold, bands, rows
            ) + fn_weight * false_negative_area(threshold, bands, rows)
            if best is None or err < best[0] - 1e-12:
                best = (err, bands, rows)
    assert best is not None
    return best[1], best[2]


def _band_groups(band: np.ndarray) -> list[np.ndarray]:
    """Groups of row indices whose band values are identical (size >= 2 only)."""
    band = np.ascontiguousarray(band)
    keys = band.view(np.dtype((np.void, band.dtype.itemsize * band.shape[1]))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    return [g for g in np.split(order, splits) if g.size > 1]


def lsh_clusters(
    signatures: np.ndarray,
    bands: int,
    rows: int,
    *,
    verify_threshold: float | None = None,
) -> np.ndarray:
    """Cluster documents whose signatures share a band; return each one's root.

    Candidates are merged with union-find, so clustering is transitive: if A
    collides with B and B with C, all three end up together even when A and C
    never collide. The root of each cluster is its smallest index, i.e. the
    first occurrence in input order, which makes "keep the root" a stable,
    order-defined policy.

    With ``verify_threshold`` set, a candidate pair is merged only if its
    estimated Jaccard reaches the threshold. This trades extra comparisons
    (quadratic in bucket size) for fewer false positives from the S-curve tail.
    """
    n_docs, num_perm = signatures.shape
    if bands * rows > num_perm:
        raise ValueError(f"bands * rows = {bands * rows} exceeds num_perm = {num_perm}")
    uf = UnionFind(n_docs)
    for band_idx in range(bands):
        cols = signatures[:, band_idx * rows : (band_idx + 1) * rows]
        for group in _band_groups(cols):
            members = group.tolist()
            if verify_threshold is None:
                head = members[0]
                for other in members[1:]:
                    uf.union(head, other)
                continue
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    if uf.find(a) == uf.find(b):
                        continue
                    if np.mean(signatures[a] == signatures[b]) >= verify_threshold:
                        uf.union(a, b)
    return np.asarray([uf.find(i) for i in range(n_docs)], dtype=np.int64)
