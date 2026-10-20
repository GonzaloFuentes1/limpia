"""The benchmark generator is part of the evidence, so it is tested too."""

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dedup_bench  # noqa: E402
import render_table  # noqa: E402


def test_corpus_is_deterministic_and_labelled():
    a = dedup_bench.generate_corpus(50, 3, seed=1)
    b = dedup_bench.generate_corpus(50, 3, seed=1)
    c = dedup_bench.generate_corpus(50, 3, seed=2)
    assert a == b and a != c
    assert len(set(a.labels)) == 50
    assert sum(r == dedup_bench.ORIGINAL for r in a.edit_rates) == 50  # one per cluster
    assert len(a.texts) == len(a.labels) == len(a.edit_rates)


def test_zero_rate_perturbation_is_identity_and_rates_reduce_similarity():
    sentences = [["la", "casa", "es", "azul"], ["el", "perro", "come", "pan", "hoy"]]
    assert dedup_bench.perturb(sentences, 0.0, random.Random(0)) == sentences
    corpus = dedup_bench.generate_corpus(150, 4, seed=0)
    pairs = dedup_bench.labelled_pairs(corpus, ngram=5)
    sims = dedup_bench.jaccard_by_level(pairs, "copy")
    values = [sims[k] for k in sorted(sims, key=float)]
    assert values[0] == 1.0
    assert values == sorted(values, reverse=True)


def test_template_pages_are_distinct_clusters_with_shared_boilerplate():
    plain = dedup_bench.generate_corpus(30, 2, seed=4)
    corpus = dedup_bench.generate_corpus(30, 2, seed=4, n_template_groups=20)
    # Adding hard negatives leaves the near-duplicate part unchanged.
    rest = [
        t for t, r in zip(corpus.texts, corpus.edit_rates, strict=True) if r != dedup_bench.TEMPLATE
    ]
    assert sorted(plain.texts) == sorted(rest)
    templated = [k for k, r in enumerate(corpus.edit_rates) if r == dedup_bench.TEMPLATE]
    assert len({corpus.labels[k] for k in templated}) == len(templated)
    pairs = dedup_bench.labelled_pairs(corpus, ngram=5)
    sims = dedup_bench.jaccard_by_level(pairs, "template")
    values = [sims[k] for k in sorted(sims, key=float)]
    assert values == sorted(values)  # more boilerplate, higher Jaccard
    assert values[0] > 0.0 and values[-1] < 1.0


def test_run_produces_consistent_results():
    payload = dedup_bench.run(
        n_clusters=40, max_copies=3, seed=0, num_perm=64, ngram=5, n_template_groups=10, repeats=2
    )
    exact, *minhash = payload["results"]
    assert exact["method"] == "exact" and exact["precision"] == 1.0
    assert exact["recall_by_edit_rate"]["0.00"] == 1.0
    assert exact["template_pairs_merged"] == 0.0
    assert len(exact["seconds_runs"]) == 2
    assert [r["threshold"] for r in minhash] == list(dedup_bench.THRESHOLDS)
    corpus = payload["corpus"]
    for row in minhash:
        assert row["bands"] * row["rows"] <= 64
        assert row["recall_by_edit_rate"]["0.00"] == 1.0
        assert row["recall"] >= exact["recall"]
        assert row["copies_at_or_above_t"] + row["copies_below_t"] == corpus["copies"]
        n_pairs = sum(b["pairs"] for b in row["scurve"])
        assert n_pairs == corpus["copies"] + corpus["template_pairs"]
    # Higher thresholds pick S-curves further to the right, so recall must not grow.
    recalls = [r["recall"] for r in minhash]
    assert recalls == sorted(recalls, reverse=True)
    table = render_table.render(payload)
    assert table.count("\n| minhash") == 3 * len(minhash)
    assert payload["hardware"]["cpu_count"] >= 1
    assert payload["hardware"]["cpu"]


def test_measured_lsh_candidates_follow_the_s_curve():
    """End-to-end check of the banding maths on pairs of known Jaccard."""
    payload = dedup_bench.run(
        n_clusters=300, max_copies=4, seed=1, num_perm=128, ngram=5, n_template_groups=80, repeats=1
    )
    checked = 0
    for row in payload["results"][1:]:
        for b in row["scurve"]:
            if b["pairs"] >= 30:
                tolerance = 4 * b["predicted_se"] + 0.03
                assert abs(b["measured"] - b["predicted"]) <= tolerance, (row["threshold"], b)
                checked += 1
    assert checked >= 10


def test_two_significant_figures():
    assert render_table.sig2(13870.4) == "14,000"
    assert render_table.sig2(1888.0) == "1,900"
    assert render_table.sig2(431.2) == "430"


def test_readme_table_is_generated_from_results_file():
    results = ROOT / "results" / "dedup_bench.json"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The markers must be present: without them this test would guard nothing.
    assert render_table.START in readme and render_table.END in readme
    payload = json.loads(results.read_text(encoding="utf-8"))
    expected = render_table.replace_section(readme, render_table.render(payload))
    assert readme == expected, "run: python benchmarks/render_table.py"
    assert np.isfinite([r["docs_per_sec"] for r in payload["results"]]).all()
