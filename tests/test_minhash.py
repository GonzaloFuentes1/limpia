"""Statistical invariants of the MinHash estimator and of LSH banding."""

import math

import numpy as np
import pytest

from limpia.dedup import MinHasher, collision_probability, estimate_jaccard, jaccard, shingles
from limpia.dedup.lsh import lsh_clusters
from limpia.dedup.minhash import EMPTY, PRIME


def _sets_with_jaccard(size: int, shift: int) -> tuple[set[str], set[str], float]:
    """Two shingle sets with exactly known Jaccard (size - shift) / (size + shift)."""
    a = {f"frase número {i}" for i in range(size)}
    b = {f"frase número {i}" for i in range(shift, size + shift)}
    return a, b, (size - shift) / (size + shift)


def test_signature_basic_properties():
    mh = MinHasher(64, seed=1)
    sig = mh.signature("el perro come pan con mantequilla todos los días")
    assert sig.dtype == np.uint32 and sig.shape == (64,)
    assert (sig < PRIME).all()
    assert (mh.signature("") == EMPTY).all()


def test_signature_matches_naive_reference():
    mh = MinHasher(32, seed=3)
    items = shingles("uno dos tres cuatro cinco seis siete ocho nueve diez once", n=3)
    hashes = mh.hash_shingles(sorted(items))
    expected = [
        min((int(a) * int(x) + int(b)) % PRIME for x in hashes)
        for a, b in zip(mh.a, mh.b, strict=True)
    ]
    assert mh.signature_from_hashes(hashes).tolist() == expected


def test_chunking_does_not_change_the_signature(monkeypatch):
    mh = MinHasher(16, seed=0)
    hashes = mh.hash_shingles(f"s{i}" for i in range(1000))
    full = mh.signature_from_hashes(hashes)
    monkeypatch.setattr("limpia.dedup.minhash._CHUNK", 7)
    assert np.array_equal(mh.signature_from_hashes(hashes), full)


@pytest.mark.parametrize(("shift", "num_perm"), [(150, 256), (67, 256), (22, 256), (67, 64)])
def test_jaccard_estimate_error_within_binomial_bound(shift, num_perm):
    """Over many seeds, |J_hat - J| stays within a Hoeffding bound.

    Each coordinate matches with probability J, so k * J_hat ~ Binomial(k, J)
    and P(|J_hat - J| > eps) <= 2 exp(-2 k eps^2). With eps chosen for a
    per-trial failure rate of delta / trials, a union bound makes the whole
    test fail with probability below delta = 1e-3 for an ideal hash family.
    The mean over seeds must also be within 4 standard errors of J
    (unbiasedness).
    """
    a, b, j_true = _sets_with_jaccard(200, shift)
    trials, delta = 150, 1e-3
    eps = math.sqrt(math.log(2 * trials / delta) / (2 * num_perm))
    proto = MinHasher(num_perm, seed=0)
    ha, hb = proto.hash_shingles(a), proto.hash_shingles(b)
    estimates = []
    for seed in range(trials):
        mh = MinHasher(num_perm, seed=seed)
        estimates.append(
            estimate_jaccard(mh.signature_from_hashes(ha), mh.signature_from_hashes(hb))
        )
    estimates = np.asarray(estimates)
    assert np.abs(estimates - j_true).max() <= eps
    stderr = math.sqrt(j_true * (1 - j_true) / (num_perm * trials))
    assert abs(estimates.mean() - j_true) <= 4 * stderr + 1e-9
    # The spread should match the binomial variance, not just be small.
    assert estimates.var() == pytest.approx(j_true * (1 - j_true) / num_perm, rel=0.35)


@pytest.mark.parametrize(
    ("shift", "bands", "rows"),
    [(150, 10, 5), (67, 10, 5), (33, 10, 5), (13, 10, 5), (67, 20, 3), (40, 8, 8)],
)
def test_lsh_candidate_probability_matches_s_curve(shift, bands, rows):
    """Empirical P(candidate) over seeds matches 1 - (1 - s^r)^b."""
    a, b, s = _sets_with_jaccard(200, shift)
    trials = 400
    proto = MinHasher(bands * rows, seed=0)
    ha, hb = proto.hash_shingles(a), proto.hash_shingles(b)
    hits = 0
    for seed in range(trials):
        mh = MinHasher(bands * rows, seed=10_000 + seed)
        sigs = np.vstack([mh.signature_from_hashes(ha), mh.signature_from_hashes(hb)])
        roots = lsh_clusters(sigs, bands, rows)
        hits += int(roots[1] == roots[0])
    p = float(collision_probability(s, bands, rows))
    tol = 4 * math.sqrt(max(p * (1 - p), 1e-4) / trials) + 0.01
    assert abs(hits / trials - p) <= tol, (s, p, hits / trials)


def test_signatures_are_deterministic_under_seed():
    texts = ["la casa es azul y grande", "la casa es azul y pequeña", "otra cosa distinta"]
    s1 = MinHasher(64, seed=42).signatures(texts)
    s2 = MinHasher(64, seed=42).signatures(texts)
    s3 = MinHasher(64, seed=43).signatures(texts)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_shingles_word_and_char():
    assert shingles("Uno, dos; TRES cuatro", n=2) == {"uno dos", "dos tres", "tres cuatro"}
    assert shingles("hola", n=5) == {"hola"}
    assert shingles("ab c", n=2, unit="char") == {"ab", "b ", " c"}
    assert shingles("   ", n=3) == set()
    with pytest.raises(ValueError):
        shingles("x", unit="byte")  # type: ignore[arg-type]
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
