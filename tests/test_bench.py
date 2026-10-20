"""The benchmark generator is part of the evidence, so it is tested too."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    assert sum(r < 0 for r in a.edit_rates) == 50  # one original per cluster
    assert len(a.texts) == len(a.labels) == len(a.edit_rates)


def test_zero_rate_perturbation_is_identity_and_rates_reduce_similarity():
    import random

    sentences = [["la", "casa", "es", "azul"], ["el", "perro", "come", "pan", "hoy"]]
    assert dedup_bench.perturb(sentences, 0.0, random.Random(0)) == sentences
    corpus = dedup_bench.generate_corpus(150, 4, seed=0)
    sims = dedup_bench.jaccard_by_edit_rate(corpus, ngram=5)
    values = [sims[k] for k in sorted(sims, key=float)]
    assert values[0] == 1.0
    assert values == sorted(values, reverse=True)


def test_run_produces_consistent_results():
    payload = dedup_bench.run(n_clusters=40, max_copies=3, seed=0, num_perm=64, ngram=5)
    exact, *minhash = payload["results"]
    assert exact["method"] == "exact" and exact["precision"] == 1.0
    assert exact["recall_by_edit_rate"]["0.00"] == 1.0
    assert [r["threshold"] for r in minhash] == list(dedup_bench.THRESHOLDS)
    for row in minhash:
        assert row["bands"] * row["rows"] <= 64
        assert row["recall_by_edit_rate"]["0.00"] == 1.0
        assert row["recall"] >= exact["recall"]
    # Higher thresholds pick S-curves further to the right, so recall must not grow.
    recalls = [r["recall"] for r in minhash]
    assert recalls == sorted(recalls, reverse=True)
    table = render_table.render(payload)
    assert table.count("\n| minhash") == 2 * len(minhash)
    assert payload["hardware"]["cpu_count"] >= 1


def test_readme_table_is_generated_from_results_file():
    results = ROOT / "results" / "dedup_bench.json"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if render_table.START not in readme:
        pytest.skip("README has no benchmark section")
    payload = json.loads(results.read_text(encoding="utf-8"))
    expected = render_table.replace_section(readme, render_table.render(payload))
    assert readme == expected, "run: python benchmarks/render_table.py"
    assert np.isfinite([r["docs_per_sec"] for r in payload["results"]]).all()
