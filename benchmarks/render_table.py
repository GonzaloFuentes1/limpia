"""Render results/dedup_bench.json as the README benchmark tables.

The README section between the markers is generated from the JSON file only,
so the numbers in the documentation cannot drift from the measured ones; a
test re-renders the tables and compares them with the README.

Usage:
    python benchmarks/render_table.py            # rewrite README.md in place
    python benchmarks/render_table.py --check    # exit 1 if README is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- dedup-bench:start -->"
END = "<!-- dedup-bench:end -->"


def sig2(x: float) -> str:
    """Round to two significant figures and print with thousands separators."""
    return f"{float(f'{x:.2g}'):,.0f}"


def _rate(value: float | None, count: int) -> str:
    return "-" if value is None else f"{value:.2f} of {count:,}"


def _docs_per_sec(row: dict, n_docs: int) -> str:
    if row["method"] == "exact":
        runs = row["seconds_runs"]
    else:
        runs = [
            a + b
            for a, b in zip(row["signature_seconds_runs"], row["cluster_seconds_runs"], strict=True)
        ]
    lo, hi = n_docs / max(runs), n_docs / min(runs)
    return f"{sig2(row['docs_per_sec'])} ({sig2(lo)}-{sig2(hi)})"


def render(payload: dict) -> str:
    hw = payload["hardware"]
    corpus = payload["corpus"]
    n_docs = corpus["documents"]
    results = payload["results"]
    minhash = [r for r in results if r["method"] == "minhash"]
    repeats = payload["timing"]["repeats"]

    lines = [
        f"Corpus: {n_docs:,} documents ({corpus['originals']:,} originals, "
        f"{corpus['copies']:,} edited copies, {corpus['template_pages']:,} templated pages in "
        f"{corpus['template_groups']:,} groups), {corpus['mean_words_per_document']:.0f} words "
        "per document on average. `J` is the true Jaccard similarity of the word 5-gram sets "
        "of a pair; a copy is merged when it ends up in its original's cluster, a template "
        "pair when its two pages end up in the same cluster.",
        "",
        "| Method | t | (b, r) | Copies, J >= t, merged | Copies, J < t, merged "
        "| Template pairs, J >= t, merged | Template pairs, J < t, merged "
        f"| Docs/s, median of {repeats} (range) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        if row["method"] == "exact":
            cells = ["exact", "-", "-", "-", "-", "-", "-"]
        else:
            cells = [
                "minhash",
                f"{row['threshold']:.1f}",
                f"({row['bands']}, {row['rows']})",
                _rate(row["copies_at_or_above_t_merged"], row["copies_at_or_above_t"]),
                _rate(row["copies_below_t_merged"], row["copies_below_t"]),
                _rate(
                    row["template_pairs_at_or_above_t_merged"],
                    row["template_pairs_at_or_above_t"],
                ),
                _rate(row["template_pairs_below_t_merged"], row["template_pairs_below_t"]),
            ]
        lines.append("| " + " | ".join([*cells, _docs_per_sec(row, n_docs)]) + " |")

    header = " | ".join(
        f"t = {r['threshold']:.1f}, (b, r) = ({r['bands']}, {r['rows']})" for r in minhash
    )
    lines += [
        "",
        "S-curve check: fraction of labelled pairs (copies and template pairs together) that "
        "share at least one LSH band, measured, next to the prediction `mean(P(J_i))` with "
        "`P(s) = 1 - (1 - s^r)^b` over the pairs in each bin. Shown as predicted / measured.",
        "",
        f"| J | Pairs | {header} |",
        "|---|---|" + "---|" * len(minhash),
    ]
    for k, bin_ in enumerate(minhash[0]["scurve"]):
        if bin_["pairs"] == 0:
            continue
        cells = [
            f"{r['scurve'][k]['predicted']:.3f} / {r['scurve'][k]['measured']:.3f}" for r in minhash
        ]
        lines.append(f"| {bin_['bin']} | {bin_['pairs']:,} | " + " | ".join(cells) + " |")

    rates = list(corpus["mean_jaccard_by_edit_rate"])
    rate_header = " | ".join(f"{float(r):.0%}" for r in rates)
    lines += [
        "",
        "Copies merged with their original, by word edit rate:",
        "",
        f"| | {rate_header} |",
        "|---|" + "---|" * len(rates),
        "| mean J | "
        + " | ".join(f"{corpus['mean_jaccard_by_edit_rate'][r]:.2f}" for r in rates)
        + " |",
    ]
    for row in results:
        label = "exact" if row["method"] == "exact" else f"minhash @ {row['threshold']:.1f}"
        lines.append(
            f"| {label} | "
            + " | ".join(f"{row['recall_by_edit_rate'][r]:.2f}" for r in rates)
            + " |"
        )
    shares = corpus["mean_jaccard_by_template_share"]
    lines += [
        "",
        "Mean J of template pairs by boilerplate share: "
        + ", ".join(f"{float(s):.0%}: {j:.2f}" for s, j in shares.items())
        + ".",
        "",
        "Pairwise scores: a pair of documents is positive if both come from the same original "
        f"({int(results[0]['true_pairs']):,} true pairs, which include copy-copy pairs); "
        "precision and recall count pairs that share a predicted cluster. Every copy counts as "
        "a duplicate however heavily it was edited, so recall here mostly reflects the edit "
        "mix of the corpus.",
        "",
        "| Method | t | Pairwise precision | Pairwise recall | Pairwise F1 |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        thr = "-" if row["method"] == "exact" else f"{row['threshold']:.1f}"
        lines.append(
            f"| {row['method']} | {thr} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} |"
        )
    lines += [
        "",
        f"Command: `{payload['command']}`. Hardware: {hw['cpu']} ({hw['cpu_count']} logical "
        f"CPUs), {hw['platform']}, Python {hw['python']}, NumPy {hw['numpy']}. Single process; "
        "MinHash throughput includes signature computation and LSH clustering. Throughput is "
        "rounded to two significant figures; the range is the slowest and fastest of the "
        "repeats.",
    ]
    return "\n".join(lines)


def replace_section(readme: str, table: str) -> str:
    start, end = readme.index(START) + len(START), readme.index(END)
    return readme[:start] + "\n" + table + "\n" + readme[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "dedup_bench.json")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    table = render(json.loads(args.results.read_text(encoding="utf-8")))
    readme = args.readme.read_text(encoding="utf-8")
    updated = replace_section(readme, table)
    if args.check:
        return int(updated != readme)
    args.readme.write_text(updated, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
