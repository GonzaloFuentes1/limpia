import random
from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np
import pytest

from fixtures import CLEAN_ES, CLEAN_PT
from limpia.dedup import (
    ExactDedupFilter,
    MinHashDedupConfig,
    MinHashDeduplicator,
    UnionFind,
    exact_clusters,
    optimal_bands_rows,
    pairwise_scores,
)
from limpia.dedup.lsh import false_negative_area, false_positive_area, lsh_clusters
from limpia.dedup.minhash import EMPTY
from limpia.document import Document
from limpia.pipeline import Pipeline

# ---------------------------------------------------------------- (b, r) choice


def _exact_fp(t: Fraction, b: int, r: int) -> Fraction:
    # P(s) = 1 - sum_k C(b,k) (-1)^k s^(rk); integrate term by term from 0 to t.
    return t - sum(comb(b, k) * (-1) ** k * t ** (r * k + 1) / (r * k + 1) for k in range(b + 1))


def _exact_fn(t: Fraction, b: int, r: int) -> Fraction:
    # 1 - P(s) = sum_k C(b,k) (-1)^k s^(rk); integrate from t to 1.
    return sum(
        comb(b, k) * (-1) ** k * (1 - t ** (r * k + 1)) / Fraction(r * k + 1) for k in range(b + 1)
    )


@pytest.mark.parametrize(("b", "r"), [(1, 1), (4, 3), (9, 13), (16, 8), (20, 6)])
@pytest.mark.parametrize("t", [Fraction(1, 2), Fraction(7, 10), Fraction(17, 20)])
def test_quadrature_matches_exact_rational_integration(b, r, t):
    assert false_positive_area(float(t), b, r) == pytest.approx(float(_exact_fp(t, b, r)), abs=1e-7)
    assert false_negative_area(float(t), b, r) == pytest.approx(float(_exact_fn(t, b, r)), abs=1e-7)


@pytest.mark.parametrize("num_perm", [8, 12, 20])
@pytest.mark.parametrize("t", [Fraction(1, 2), Fraction(7, 10), Fraction(17, 20)])
def test_optimal_bands_rows_matches_exact_brute_force(num_perm, t):
    errors = {
        (b, r): (_exact_fp(t, b, r) + _exact_fn(t, b, r)) / 2
        for b in range(1, num_perm + 1)
        for r in range(1, num_perm // b + 1)
    }
    best = min(errors.values())
    b, r = optimal_bands_rows(float(t), num_perm)
    assert b * r <= num_perm
    # Exact optimum, up to floating point ties.
    assert float(errors[(b, r)] - best) < 1e-9


def test_optimal_bands_rows_moves_the_s_curve_with_the_threshold():
    knees = []
    for t in (0.5, 0.7, 0.9):
        b, r = optimal_bands_rows(t, 128)
        knees.append((1 / b) ** (1 / r))
    assert knees == sorted(knees)
    with pytest.raises(ValueError):
        optimal_bands_rows(1.0, 128)


# ------------------------------------------------------------------- union-find


@pytest.mark.parametrize("seed", range(10))
def test_union_find_matches_bfs_components(seed):
    rng = random.Random(seed)
    n = 60
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 80))]
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    for start in range(n):
        seen, frontier = {start}, [start]
        while frontier:
            nxt = [v for u in frontier for v in adj[u] if v not in seen]
            seen.update(nxt)
            frontier = nxt
        assert uf.find(start) == min(seen)


# --------------------------------------------------------------------- metrics


@pytest.mark.parametrize("seed", range(10))
def test_pairwise_scores_match_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 40)
    pred = [rng.randrange(6) for _ in range(n)]
    true = [rng.randrange(6) for _ in range(n)]
    pairs = list(combinations(range(n), 2))
    pp = {p for p in pairs if pred[p[0]] == pred[p[1]]}
    tp = {p for p in pairs if true[p[0]] == true[p[1]]}
    got = pairwise_scores(pred, true)
    assert got["precision"] == pytest.approx(len(pp & tp) / len(pp) if pp else 1.0)
    assert got["recall"] == pytest.approx(len(pp & tp) / len(tp) if tp else 1.0)


# ----------------------------------------------------------------- exact dedup


def test_exact_dedup_ignores_case_punctuation_and_whitespace():
    texts = [CLEAN_ES, CLEAN_ES.upper().replace(",", ""), "  " + CLEAN_ES + "\n\n", CLEAN_PT]
    assert exact_clusters(texts).tolist() == [0, 0, 0, 3]
    pipe = Pipeline([ExactDedupFilter()])
    kept = list(pipe.run(Document(str(i), t) for i, t in enumerate(texts)))
    assert [d.id for d in kept] == ["0", "3"]
    assert pipe.report()["steps"][0]["reasons"] == {"exact_dedup:duplicate": 2}


def test_exact_dedup_keeps_accent_differences():
    assert exact_clusters(["Esta es mi casa.", "Está es mi casa."]).tolist() == [0, 1]


# Texts that normalize to the empty string: nothing is left to compare.
EMPTY_AFTER_NORMALIZATION = ["", "   ", "!!! ???", "★★★", "😀😀", "— « » —"]


def test_exact_dedup_keeps_texts_that_normalize_to_nothing():
    texts = [*EMPTY_AFTER_NORMALIZATION, "Hola.", "hola"]
    n = len(EMPTY_AFTER_NORMALIZATION)
    assert exact_clusters(texts).tolist() == [*range(n), n, n]
    pipe = Pipeline([ExactDedupFilter()])
    kept = list(pipe.run(Document(str(i), t) for i, t in enumerate(texts)))
    assert len(kept) == n + 1


def test_near_dedup_keeps_texts_with_no_shingles_as_singletons():
    texts = [*EMPTY_AFTER_NORMALIZATION, CLEAN_ES, CLEAN_ES.upper()]
    dd = MinHashDeduplicator(MinHashDedupConfig(threshold=0.8, num_perm=64, seed=0))
    sigs = dd.signatures(texts)
    n = len(EMPTY_AFTER_NORMALIZATION)
    assert np.all(sigs[:n] == EMPTY)
    for verify in (False, True):
        roots = lsh_clusters(sigs, dd.bands, dd.rows, verify_threshold=0.8 if verify else None)
        assert roots.tolist() == [*range(n), n, n]


# ------------------------------------------------------------------ near dedup


def _perturb(text: str, rng: random.Random, n_edits: int) -> str:
    words = text.split()
    for _ in range(n_edits):
        words[rng.randrange(len(words))] = rng.choice(["casa", "río", "mesa", "libro"])
    return " ".join(words)


def test_near_duplicates_cluster_and_first_occurrence_is_kept():
    rng = random.Random(0)
    docs = [
        Document("es", CLEAN_ES),
        Document("pt", CLEAN_PT),
        Document("es-copy", _perturb(CLEAN_ES, rng, 2)),
        Document("pt-copy", _perturb(CLEAN_PT, rng, 1)),
        Document("es-copy2", _perturb(CLEAN_ES, rng, 1)),
    ]
    dd = MinHashDeduplicator(MinHashDedupConfig(threshold=0.7, num_perm=128, seed=1))
    kept, roots = dd.deduplicate(docs)
    assert [d.id for d in kept] == ["es", "pt"]
    assert roots.tolist() == [0, 1, 0, 1, 0]


@pytest.mark.parametrize("verify", [False, True])
def test_near_dedup_is_deterministic_and_respects_threshold(verify):
    rng = random.Random(5)
    base = [_perturb(CLEAN_ES, rng, 40) for _ in range(5)]  # mutually dissimilar
    texts = base + [_perturb(t, rng, 1) for t in base]
    cfg = MinHashDedupConfig(threshold=0.8, num_perm=128, seed=3, verify=verify)
    roots_1 = MinHashDeduplicator(cfg).cluster(MinHashDeduplicator(cfg).signatures(texts))
    roots_2 = MinHashDeduplicator(cfg).cluster(MinHashDeduplicator(cfg).signatures(texts))
    assert np.array_equal(roots_1, roots_2)
    assert roots_1.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_clustering_is_transitive():
    # Bands 0 links A-B, band 1 links B-C; A and C share nothing directly.
    sigs = np.array([[1, 1, 9, 9], [1, 1, 2, 2], [7, 7, 2, 2], [5, 5, 6, 6]], dtype=np.uint32)
    assert lsh_clusters(sigs, bands=2, rows=2).tolist() == [0, 0, 0, 3]
    # Verification rejects pairs whose estimated Jaccard is below the threshold.
    assert lsh_clusters(sigs, bands=2, rows=2, verify_threshold=0.75).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        lsh_clusters(sigs, bands=3, rows=2)


def test_explicit_bands_rows_require_both():
    with pytest.raises(ValueError):
        MinHashDeduplicator(MinHashDedupConfig(bands=4))
    dd = MinHashDeduplicator(MinHashDedupConfig(num_perm=32, bands=4, rows=8))
    assert (dd.bands, dd.rows) == (4, 8)
