"""Render results/dedup_bench.json as the README benchmark table.

The README section between the markers is generated from the JSON file only,
so the numbers in the documentation cannot drift from the measured ones; a
test re-renders the table and compares it with the README.

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


def render(payload: dict) -> str:
    hw = payload["hardware"]
    corpus = payload["corpus"]
    lines = [
        "| Method | Threshold | (b, r) | Precision | Recall | F1 | Docs/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["results"]:
        if row["method"] == "exact":
            thr, br = "-", "-"
        else:
            thr, br = f"{row['threshold']:.1f}", f"({row['bands']}, {row['rows']})"
        lines.append(
            f"| {row['method']} | {thr} | {br} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['docs_per_sec']:,.0f} |"
        )
    rates = list(corpus["mean_jaccard_by_edit_rate"])
    header = " | ".join(f"{float(r):.0%}" for r in rates)
    lines += [
        "",
        f"Recall of copies against their original, by word edit rate ({corpus['copies']:,} "
        f"copies of {corpus['clusters']:,} originals, {corpus['documents']:,} documents):",
        "",
        f"| | {header} |",
        "|---|" + "---|" * len(rates),
        "| mean true Jaccard (word 5-grams) | "
        + " | ".join(f"{corpus['mean_jaccard_by_edit_rate'][r]:.2f}" for r in rates)
        + " |",
    ]
    for row in payload["results"]:
        label = "exact" if row["method"] == "exact" else f"minhash @ {row['threshold']:.1f}"
        lines.append(
            f"| {label} | "
            + " | ".join(f"{row['recall_by_edit_rate'][r]:.2f}" for r in rates)
            + " |"
        )
    lines += [
        "",
        f"Command: `{payload['command']}`. Hardware: {hw['processor']} "
        f"({hw['cpu_count']} logical CPUs), {hw['platform']}, Python {hw['python']}, "
        f"NumPy {hw['numpy']}. Single process; MinHash throughput includes signature "
        "computation and LSH clustering.",
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
