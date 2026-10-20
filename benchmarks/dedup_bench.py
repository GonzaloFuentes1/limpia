"""Exact vs MinHash deduplication on a synthetic Spanish corpus with known Jaccard.

The corpus has two parts, both seeded:

* Near-duplicates. Each original gets up to ``max_copies`` perturbed copies
  (word insertions, deletions, substitutions and adjacent swaps at a rate drawn
  from a few levels, plus a sentence swap). A copy is labelled with its
  original's cluster however heavily it was edited.
* Hard negatives. Groups of templated pages share header and footer
  boilerplate and differ in their page-specific sentences. Every templated page
  is its own cluster, so merging two of them is a false positive even when
  their word 5-gram Jaccard is high.

For every labelled pair (copy and original, or two pages of the same template)
the true Jaccard ``J`` of the word 5-gram sets is computed. The benchmark then
reports, for each MinHash threshold ``t``: how often pairs with ``J >= t`` and
``J < t`` end up in the same cluster, how often template pairs are merged, and,
per bin of ``J``, the measured rate of direct LSH candidates next to the
S-curve prediction ``mean(P(J_i))`` with ``P(s) = 1 - (1 - s**r)**b``.
Pairwise precision, recall and F1 over cluster co-membership are reported
too. Throughput is the median of several timed repeats.

Usage:
    python benchmarks/dedup_bench.py --out results/dedup_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from limpia import __version__
from limpia.dedup import (
    MinHashDedupConfig,
    MinHashDeduplicator,
    collision_probability,
    exact_clusters,
    jaccard,
    pairwise_scores,
    shingles,
)

# Common Spanish words; the generator samples them with Zipf-like weights so
# that function words dominate, as in real text.
_WORDS = """
de la que el en y a los se del las un por con no una su para es al lo como más pero sus le
ya o fue este ha sí porque esta son entre está cuando muy sin sobre ser tiene también me hasta
hay donde han quien están desde todo nos durante estados todos uno les ni contra otros fueron
ese eso había ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa
estos mucho quienes nada muchos cual sea poco ella estar haber estas estaba estamos algunas
algo nosotros gobierno país ciudad región comuna municipio alcalde ministra ministro proyecto
ley salud educación escuela liceo universidad estudiantes profesores familia familias niños
trabajo trabajadores empresa empresas mercado precio precios agua luz camino calle barrio
vecinos junta reunión consejo plan programa servicio servicios hospital médico pacientes
invierno verano otoño primavera lluvia sequía río mar costa cordillera valle campo cosecha
agricultores pescadores minería cobre litio energía viento sol paneles red redes internet
datos informe estudio encuesta resultados aumento baja crecimiento economía inflación
sueldo pensiones vivienda arriendo subsidio postulación beneficio apoyo seguridad policía
tribunal justicia derecho derechos constitución elección elecciones votos candidato
candidata partido debate prensa radio diario noticia noticias historia cultura música
fútbol partido equipo campeonato estadio hinchas jugador entrenador temporada torneo
semana mes año años día días hora horas mañana tarde noche lunes martes miércoles jueves
viernes sábado domingo nuevo nueva gran grande pequeño pequeña público pública social
nacional regional local rural urbano importante principal mejor peor primer primera último
última según además sin embargo aunque mientras luego después antes siempre nunca hoy ayer
anunció dijo explicó señaló afirmó indicó informó aseguró destacó agregó recordó propuso
aprobó rechazó presentó comenzó terminó llegó salió vivió trabaja vive necesita puede deben
"""
VOCAB = _WORDS.split()

SENTENCE_LEN = (8, 20)
DOC_SENTENCES = (4, 10)
EDIT_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20)
THRESHOLDS = (0.6, 0.7, 0.8)
# Fraction of a templated page's words that are shared boilerplate.
TEMPLATE_SHARES = (0.5, 0.7, 0.85, 0.95)
TEMPLATE_WORDS = (100, 160)
TEMPLATE_PAGES = (2, 4)
JACCARD_BINS = tuple(round(0.1 * i, 1) for i in range(11))

ORIGINAL = -1.0
TEMPLATE = -2.0


@dataclass(frozen=True)
class Corpus:
    texts: list[str]
    labels: list[int]  # ground-truth cluster id per document
    edit_rates: list[float]  # ORIGINAL, TEMPLATE, or the copy's word edit rate
    template_groups: list[int]  # template group per document, -1 if not templated
    template_shares: list[float]  # boilerplate share of the group, -1.0 if not templated


def _weights() -> list[float]:
    return [1.0 / (rank + 1) ** 1.1 for rank in range(len(VOCAB))]


def _sentence(rng: random.Random, weights: list[float]) -> list[str]:
    n = rng.randint(*SENTENCE_LEN)
    return rng.choices(VOCAB, weights=weights, k=n)


def _words_to_sentences(words: list[str], rng: random.Random) -> list[list[str]]:
    out, i = [], 0
    while i < len(words):
        n = rng.randint(*SENTENCE_LEN)
        out.append(words[i : i + n])
        i += n
    return out


def _render(sentences: list[list[str]]) -> str:
    return " ".join(" ".join(s).capitalize() + "." for s in sentences if s)


def perturb(sentences: list[list[str]], rate: float, rng: random.Random) -> list[list[str]]:
    """Apply ``round(rate * n_words)`` random word edits and maybe a sentence swap."""
    out = [list(s) for s in sentences]
    n_words = sum(len(s) for s in out)
    for _ in range(round(rate * n_words)):
        s = out[rng.randrange(len(out))]
        op = rng.choice(("insert", "delete", "substitute", "swap"))
        i = rng.randrange(len(s))
        if op == "insert":
            s.insert(i, rng.choice(VOCAB))
        elif op == "delete" and len(s) > 1:
            del s[i]
        elif op == "substitute":
            s[i] = rng.choice(VOCAB)
        elif op == "swap" and i + 1 < len(s):
            s[i], s[i + 1] = s[i + 1], s[i]
    if rate > 0 and len(out) > 1 and rng.random() < 0.5:
        a, b = rng.sample(range(len(out)), 2)
        out[a], out[b] = out[b], out[a]
    return out


def templated_pages(share: float, n_pages: int, rng: random.Random) -> list[str]:
    """Pages that share header and footer boilerplate and differ in the middle."""
    weights = _weights()
    n_words = rng.randint(*TEMPLATE_WORDS)
    n_shared = round(share * n_words)
    shared = rng.choices(VOCAB, weights=weights, k=n_shared)
    header = _words_to_sentences(shared[: n_shared // 2], rng)
    footer = _words_to_sentences(shared[n_shared // 2 :], rng)
    pages = []
    for _ in range(n_pages):
        own = rng.choices(VOCAB, weights=weights, k=n_words - n_shared)
        pages.append(_render(header + _words_to_sentences(own, rng) + footer))
    return pages


def generate_corpus(
    n_clusters: int, max_copies: int, seed: int, n_template_groups: int = 0
) -> Corpus:
    rng = random.Random(seed)
    weights = _weights()
    rows: list[tuple[str, int, float, int, float]] = []
    for cluster in range(n_clusters):
        sentences = [_sentence(rng, weights) for _ in range(rng.randint(*DOC_SENTENCES))]
        rows.append((_render(sentences), cluster, ORIGINAL, -1, -1.0))
        for _ in range(rng.randint(0, max_copies)):
            rate = rng.choice(EDIT_LEVELS)
            rows.append((_render(perturb(sentences, rate, rng)), cluster, rate, -1, -1.0))
    # Hard negatives come from an independent stream so that adding them does
    # not change the near-duplicate part of the corpus.
    trng = random.Random(seed + 1_000_003)
    label = n_clusters
    for group in range(n_template_groups):
        share = trng.choice(TEMPLATE_SHARES)
        for page in templated_pages(share, trng.randint(*TEMPLATE_PAGES), trng):
            rows.append((page, label, TEMPLATE, group, share))
            label += 1
    rng.shuffle(rows)
    return Corpus(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        [r[4] for r in rows],
    )


@dataclass(frozen=True)
class Pairs:
    """Labelled pairs with their true Jaccard similarity."""

    i: np.ndarray
    j: np.ndarray
    kind: list[str]  # "copy" or "template"
    level: list[float]  # copy edit rate or template boilerplate share
    jaccard: np.ndarray


def labelled_pairs(corpus: Corpus, ngram: int) -> Pairs:
    """Copy-original pairs and all pairs of pages within a template group."""
    original_idx = {
        label: k
        for k, (label, rate) in enumerate(zip(corpus.labels, corpus.edit_rates, strict=True))
        if rate == ORIGINAL
    }
    groups: dict[int, list[int]] = {}
    for k, g in enumerate(corpus.template_groups):
        if g >= 0:
            groups.setdefault(g, []).append(k)
    ii, jj, kind, level = [], [], [], []
    for k, (label, rate) in enumerate(zip(corpus.labels, corpus.edit_rates, strict=True)):
        if rate >= 0:
            ii.append(original_idx[label])
            jj.append(k)
            kind.append("copy")
            level.append(rate)
    for members in groups.values():
        for pos, a in enumerate(members):
            for b in members[pos + 1 :]:
                ii.append(a)
                jj.append(b)
                kind.append("template")
                level.append(corpus.template_shares[a])
    cache: dict[int, set[str]] = {}

    def sh(k: int) -> set[str]:
        if k not in cache:
            cache[k] = shingles(corpus.texts[k], ngram)
        return cache[k]

    sims = np.array([jaccard(sh(a), sh(b)) for a, b in zip(ii, jj, strict=True)])
    return Pairs(np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64), kind, level, sims)


def direct_candidates(sigs: np.ndarray, pairs: Pairs, bands: int, rows: int) -> np.ndarray:
    """Whether each pair shares at least one identical band (an LSH candidate)."""
    a, b = sigs[pairs.i, : bands * rows], sigs[pairs.j, : bands * rows]
    equal = (a == b).reshape(len(pairs.i), bands, rows)
    return equal.all(axis=2).any(axis=1)


def _rate(mask: np.ndarray) -> float | None:
    return float(np.mean(mask)) if mask.size else None


def scurve_check(
    pairs: Pairs, candidates: np.ndarray, bands: int, rows: int
) -> list[dict[str, object]]:
    """Measured direct-candidate rate vs the S-curve prediction, per Jaccard bin.

    ``predicted_se`` is the standard deviation of the measured rate if each
    pair were an independent Bernoulli(P(J_i)) trial.
    """
    predicted = collision_probability(pairs.jaccard, bands, rows)
    out = []
    edges = JACCARD_BINS
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        upper = (pairs.jaccard <= hi) if k == len(edges) - 2 else (pairs.jaccard < hi)
        mask = (pairs.jaccard >= lo) & upper
        n = int(mask.sum())
        p = predicted[mask]
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "pairs": n,
                "predicted": float(p.mean()) if n else None,
                "predicted_se": float(np.sqrt((p * (1 - p)).sum()) / n) if n else None,
                "measured": _rate(candidates[mask]),
            }
        )
    return out


def recall_by_edit_rate(roots: np.ndarray, pairs: Pairs) -> dict[str, float]:
    """Fraction of copies at each edit rate that land in their original's cluster."""
    together = roots[pairs.i] == roots[pairs.j]
    out: dict[str, float] = {}
    for rate in EDIT_LEVELS:
        mask = np.array(
            [k == "copy" and lv == rate for k, lv in zip(pairs.kind, pairs.level, strict=True)]
        )
        if mask.any():
            out[f"{rate:.2f}"] = float(together[mask].mean())
    return out


def jaccard_by_level(pairs: Pairs, kind: str) -> dict[str, float]:
    levels = sorted({lv for k, lv in zip(pairs.kind, pairs.level, strict=True) if k == kind})
    out = {}
    for lv in levels:
        mask = np.array(
            [k == kind and x == lv for k, x in zip(pairs.kind, pairs.level, strict=True)]
        )
        out[f"{lv:.2f}"] = float(pairs.jaccard[mask].mean())
    return out


def cpu_model() -> str:
    """Human-readable CPU model name, or the platform's processor string."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                return m.group(1).strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def hardware() -> dict[str, object]:
    return {
        "cpu": cpu_model(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _timed(fn, repeats: int) -> tuple[float, list[float]]:
    """Median and all wall-clock times of ``repeats`` calls of ``fn``."""
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs), runs


def run(
    n_clusters: int,
    max_copies: int,
    seed: int,
    num_perm: int,
    ngram: int,
    n_template_groups: int = 300,
    repeats: int = 5,
) -> dict:
    corpus = generate_corpus(n_clusters, max_copies, seed, n_template_groups)
    pairs = labelled_pairs(corpus, ngram)
    n = len(corpus.texts)
    n_words = sum(len(t.split()) for t in corpus.texts)
    is_copy = np.array([k == "copy" for k in pairs.kind], dtype=bool)
    is_template = ~is_copy
    results = []

    roots = exact_clusters(corpus.texts)
    seconds, runs = _timed(lambda: exact_clusters(corpus.texts), repeats)
    together = roots[pairs.i] == roots[pairs.j]
    results.append(
        {
            "method": "exact",
            **pairwise_scores(roots.tolist(), corpus.labels),
            "recall_by_edit_rate": recall_by_edit_rate(roots, pairs),
            "template_pairs_merged": _rate(together[is_template]),
            "seconds_median": seconds,
            "seconds_runs": runs,
            "docs_per_sec": n / seconds,
            "words_per_sec": n_words / seconds,
        }
    )

    base = MinHashDeduplicator(MinHashDedupConfig(num_perm=num_perm, ngram=ngram, seed=seed))
    sigs = base.signatures(corpus.texts)
    sig_seconds, sig_runs = _timed(lambda: base.signatures(corpus.texts), repeats)
    for threshold in THRESHOLDS:
        dd = MinHashDeduplicator(
            MinHashDedupConfig(threshold=threshold, num_perm=num_perm, ngram=ngram, seed=seed)
        )
        roots = dd.cluster(sigs)
        cluster_seconds, cluster_runs = _timed(lambda dd=dd: dd.cluster(sigs), repeats)
        together = roots[pairs.i] == roots[pairs.j]
        above = pairs.jaccard >= threshold
        candidates = direct_candidates(sigs, pairs, dd.bands, dd.rows)
        total = sig_seconds + cluster_seconds
        results.append(
            {
                "method": "minhash",
                "threshold": threshold,
                "bands": dd.bands,
                "rows": dd.rows,
                **pairwise_scores(roots.tolist(), corpus.labels),
                "recall_by_edit_rate": recall_by_edit_rate(roots, pairs),
                "copies_at_or_above_t": int((is_copy & above).sum()),
                "copies_at_or_above_t_merged": _rate(together[is_copy & above]),
                "copies_below_t": int((is_copy & ~above).sum()),
                "copies_below_t_merged": _rate(together[is_copy & ~above]),
                "template_pairs_at_or_above_t": int((is_template & above).sum()),
                "template_pairs_at_or_above_t_merged": _rate(together[is_template & above]),
                "template_pairs_below_t": int((is_template & ~above).sum()),
                "template_pairs_below_t_merged": _rate(together[is_template & ~above]),
                "template_pairs_merged": _rate(together[is_template]),
                "scurve": scurve_check(pairs, candidates, dd.bands, dd.rows),
                "seconds_median": total,
                "signature_seconds_runs": sig_runs,
                "cluster_seconds_runs": cluster_runs,
                "docs_per_sec": n / total,
                "words_per_sec": n_words / total,
            }
        )

    return {
        "limpia_version": __version__,
        "command": "python benchmarks/dedup_bench.py "
        f"--n-clusters {n_clusters} --max-copies {max_copies} --seed {seed} "
        f"--num-perm {num_perm} --ngram {ngram} --n-template-groups {n_template_groups} "
        f"--repeats {repeats}",
        "corpus": {
            "documents": n,
            "mean_words_per_document": n_words / n,
            "clusters": n_clusters,
            "originals": sum(r == ORIGINAL for r in corpus.edit_rates),
            "copies": sum(r >= 0 for r in corpus.edit_rates),
            "template_groups": n_template_groups,
            "template_pages": sum(r == TEMPLATE for r in corpus.edit_rates),
            "template_pairs": int(is_template.sum()),
            "edit_levels": list(EDIT_LEVELS),
            "template_shares": list(TEMPLATE_SHARES),
            "mean_jaccard_by_edit_rate": jaccard_by_level(pairs, "copy"),
            "mean_jaccard_by_template_share": jaccard_by_level(pairs, "template"),
        },
        "params": {"num_perm": num_perm, "ngram": ngram, "unit": "word", "seed": seed},
        "timing": {"repeats": repeats, "statistic": "median"},
        "hardware": hardware(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-clusters", type=int, default=2000)
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--n-template-groups", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/dedup_bench.json"))
    args = parser.parse_args()
    payload = run(
        args.n_clusters,
        args.max_copies,
        args.seed,
        args.num_perm,
        args.ngram,
        args.n_template_groups,
        args.repeats,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    for row in payload["results"]:
        label = row["method"] + (f"@{row['threshold']}" if "threshold" in row else "")
        print(
            f"{label:12s} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} {row['docs_per_sec']:.0f} docs/s"
        )


if __name__ == "__main__":
    main()
