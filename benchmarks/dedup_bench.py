"""Exact vs MinHash deduplication on a synthetic Spanish near-duplicate corpus.

The corpus has known ground truth: every document is either an original or a
perturbed copy of one, and the copy belongs to its original's cluster no
matter how heavily it was edited. Perturbations are seeded word insertions,
deletions, substitutions and adjacent swaps, plus sentence swaps; their rate
is drawn per copy from a few levels so that the corpus contains easy and hard
near-duplicates. Pairwise precision/recall/F1 is then measured for exact
dedup and for MinHash LSH at several thresholds.

Usage:
    python benchmarks/dedup_bench.py --out results/dedup_bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from limpia import __version__
from limpia.dedup import (
    MinHashDedupConfig,
    MinHashDeduplicator,
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


@dataclass(frozen=True)
class Corpus:
    texts: list[str]
    labels: list[int]  # ground-truth cluster id per document
    edit_rates: list[float]  # -1.0 for originals


def _sentence(rng: random.Random, weights: list[float]) -> list[str]:
    n = rng.randint(*SENTENCE_LEN)
    return rng.choices(VOCAB, weights=weights, k=n)


def _render(sentences: list[list[str]]) -> str:
    return " ".join(" ".join(s).capitalize() + "." for s in sentences)


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


def generate_corpus(n_clusters: int, max_copies: int, seed: int) -> Corpus:
    rng = random.Random(seed)
    weights = [1.0 / (rank + 1) ** 1.1 for rank in range(len(VOCAB))]
    rows: list[tuple[str, int, float]] = []
    for cluster in range(n_clusters):
        sentences = [_sentence(rng, weights) for _ in range(rng.randint(*DOC_SENTENCES))]
        rows.append((_render(sentences), cluster, -1.0))
        for _ in range(rng.randint(0, max_copies)):
            rate = rng.choice(EDIT_LEVELS)
            rows.append((_render(perturb(sentences, rate, rng)), cluster, rate))
    rng.shuffle(rows)
    return Corpus([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def jaccard_by_edit_rate(corpus: Corpus, ngram: int) -> dict[str, float]:
    """Mean true Jaccard between each copy and its original, per edit rate."""
    originals = {
        label: text
        for text, label, rate in zip(corpus.texts, corpus.labels, corpus.edit_rates, strict=True)
        if rate < 0
    }
    sims: dict[float, list[float]] = {}
    for text, label, rate in zip(corpus.texts, corpus.labels, corpus.edit_rates, strict=True):
        if rate >= 0:
            j = jaccard(shingles(text, ngram), shingles(originals[label], ngram))
            sims.setdefault(rate, []).append(j)
    return {f"{rate:.2f}": float(np.mean(v)) for rate, v in sorted(sims.items())}


def recall_by_edit_rate(corpus: Corpus, roots: np.ndarray) -> dict[str, float]:
    """Fraction of copies at each edit rate that land in their original's cluster."""
    original_idx = {
        label: i
        for i, (label, rate) in enumerate(zip(corpus.labels, corpus.edit_rates, strict=True))
        if rate < 0
    }
    hits: dict[float, list[bool]] = {}
    for i, (label, rate) in enumerate(zip(corpus.labels, corpus.edit_rates, strict=True)):
        if rate >= 0:
            hits.setdefault(rate, []).append(bool(roots[i] == roots[original_idx[label]]))
    return {f"{rate:.2f}": float(np.mean(v)) for rate, v in sorted(hits.items())}


def hardware() -> dict[str, object]:
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def run(n_clusters: int, max_copies: int, seed: int, num_perm: int, ngram: int) -> dict:
    corpus = generate_corpus(n_clusters, max_copies, seed)
    n = len(corpus.texts)
    results = []

    t0 = time.perf_counter()
    roots = exact_clusters(corpus.texts)
    elapsed = time.perf_counter() - t0
    results.append(
        {
            "method": "exact",
            **pairwise_scores(roots.tolist(), corpus.labels),
            "recall_by_edit_rate": recall_by_edit_rate(corpus, roots),
            "seconds": elapsed,
            "docs_per_sec": n / elapsed,
        }
    )

    base = MinHashDeduplicator(MinHashDedupConfig(num_perm=num_perm, ngram=ngram, seed=seed))
    t0 = time.perf_counter()
    sigs = base.signatures(corpus.texts)
    sig_seconds = time.perf_counter() - t0
    for threshold in THRESHOLDS:
        dd = MinHashDeduplicator(
            MinHashDedupConfig(threshold=threshold, num_perm=num_perm, ngram=ngram, seed=seed)
        )
        t0 = time.perf_counter()
        roots = dd.cluster(sigs)
        cluster_seconds = time.perf_counter() - t0
        total = sig_seconds + cluster_seconds
        results.append(
            {
                "method": "minhash",
                "threshold": threshold,
                "bands": dd.bands,
                "rows": dd.rows,
                **pairwise_scores(roots.tolist(), corpus.labels),
                "recall_by_edit_rate": recall_by_edit_rate(corpus, roots),
                "seconds": total,
                "signature_seconds": sig_seconds,
                "docs_per_sec": n / total,
            }
        )

    return {
        "limpia_version": __version__,
        "command": "python benchmarks/dedup_bench.py "
        f"--n-clusters {n_clusters} --max-copies {max_copies} --seed {seed} "
        f"--num-perm {num_perm} --ngram {ngram}",
        "corpus": {
            "documents": n,
            "clusters": n_clusters,
            "originals": sum(r < 0 for r in corpus.edit_rates),
            "copies": sum(r >= 0 for r in corpus.edit_rates),
            "edit_levels": list(EDIT_LEVELS),
            "mean_jaccard_by_edit_rate": jaccard_by_edit_rate(corpus, ngram),
        },
        "params": {"num_perm": num_perm, "ngram": ngram, "unit": "word", "seed": seed},
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
    parser.add_argument("--out", type=Path, default=Path("results/dedup_bench.json"))
    args = parser.parse_args()
    payload = run(args.n_clusters, args.max_copies, args.seed, args.num_perm, args.ngram)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in payload["results"]:
        label = row["method"] + (f"@{row['threshold']}" if "threshold" in row else "")
        print(
            f"{label:12s} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} {row['docs_per_sec']:.0f} docs/s"
        )


if __name__ == "__main__":
    main()
