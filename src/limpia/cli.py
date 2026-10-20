"""Command-line interface over JSONL files.

Two commands mirror the two execution models:

* ``limpia run`` streams documents through per-document steps (quality
  filters, exact dedup, decontamination, PII scrubbing) in one pass.
* ``limpia dedup`` needs global state, so it makes two passes: the first
  computes MinHash signatures (kept in memory, ``4 * num_perm`` bytes per
  document), the second re-reads the input and writes cluster representatives.
  Re-reading avoids holding the texts themselves in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from limpia import __version__
from limpia.config import load_language_config
from limpia.decontam import ContaminationPolicy, NGramIndex, decontamination_step
from limpia.dedup import ExactDedupFilter, MinHashDedupConfig, MinHashDeduplicator
from limpia.document import Document
from limpia.filters import quality_filters
from limpia.io import read_jsonl, write_jsonl
from limpia.pii import PIIScrubber
from limpia.pipeline import Pipeline, Step


def _benchmark_texts(path: Path, text_key: str) -> Iterator[str]:
    if path.suffix == ".txt":
        yield from (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        yield from (d.text for d in read_jsonl(path, text_key=text_key))


def _write_json(payload: dict, path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    steps: list[Step] = []
    if not args.no_quality:
        steps.extend(quality_filters(load_language_config(args.lang)))
    if args.exact_dedup:
        steps.append(ExactDedupFilter())
    if args.decontam:
        index = NGramIndex.from_texts(
            _benchmark_texts(Path(args.decontam), args.text_key), n=args.decontam_n
        )
        policy = ContaminationPolicy(args.decontam_min_matches, args.decontam_min_fraction)
        steps.append(decontamination_step(index, args.decontam_mode, policy))
    if args.pii:
        steps.append(PIIScrubber())
    if not steps:
        print("error: no steps selected", file=sys.stderr)
        return 2
    pipe = Pipeline(steps)
    docs = read_jsonl(args.input, text_key=args.text_key, id_key=args.id_key)
    write_jsonl(pipe.run(docs), args.output, text_key=args.text_key, id_key=args.id_key)
    report = pipe.report()
    _write_json(report, Path(args.report) if args.report else None)
    print(
        f"kept {report['output_docs']}/{report['input_docs']} documents -> {args.output}",
        file=sys.stderr,
    )
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    cfg = MinHashDedupConfig(
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram=args.ngram,
        unit=args.unit,
        seed=args.seed,
        verify=args.verify,
    )
    dd = MinHashDeduplicator(cfg)

    def read() -> Iterator[Document]:
        return read_jsonl(args.input, text_key=args.text_key, id_key=args.id_key)

    roots = dd.cluster(dd.signatures(d.text for d in read()))
    kept = (d for i, d in enumerate(read()) if roots[i] == i)
    n_out = write_jsonl(kept, args.output, text_key=args.text_key, id_key=args.id_key)
    report = {
        "limpia_version": __version__,
        "input_docs": len(roots),
        "output_docs": n_out,
        "removed": len(roots) - n_out,
        "clusters_with_duplicates": len({int(r) for i, r in enumerate(roots) if r != i}),
        "config": {**cfg.__dict__, "bands": dd.bands, "rows": dd.rows},
    }
    _write_json(report, Path(args.report) if args.report else None)
    print(f"kept {n_out}/{len(roots)} documents -> {args.output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="limpia", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"limpia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def io_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", help="input JSONL (.jsonl or .jsonl.gz)")
        p.add_argument("output", help="output JSONL (.jsonl or .jsonl.gz)")
        p.add_argument("--report", help="write the JSON report here (default: stdout)")
        p.add_argument("--text-key", default="text")
        p.add_argument("--id-key", default="id")

    run = sub.add_parser("run", help="stream documents through filters and transforms")
    io_args(run)
    run.add_argument(
        "--lang",
        default="es",
        help="bundled config (es, pt, es-fineweb2, pt-fineweb2) or a YAML path",
    )
    run.add_argument("--no-quality", action="store_true", help="skip the quality filters")
    run.add_argument("--exact-dedup", action="store_true", help="drop exact duplicates")
    run.add_argument("--pii", action="store_true", help="replace PII with typed tags")
    run.add_argument("--decontam", help="benchmark texts (.jsonl or .txt, one per line)")
    run.add_argument("--decontam-n", type=int, default=13)
    run.add_argument("--decontam-mode", choices=["drop", "flag"], default="drop")
    run.add_argument("--decontam-min-matches", type=int, default=1)
    run.add_argument("--decontam-min-fraction", type=float, default=0.0)
    run.set_defaults(func=cmd_run)

    dedup = sub.add_parser("dedup", help="MinHash LSH near-duplicate removal (two passes)")
    io_args(dedup)
    dedup.add_argument("--threshold", type=float, default=0.8)
    dedup.add_argument("--num-perm", type=int, default=128)
    dedup.add_argument("--ngram", type=int, default=5)
    dedup.add_argument("--unit", choices=["word", "char"], default="word")
    dedup.add_argument("--seed", type=int, default=0)
    dedup.add_argument("--verify", action="store_true", help="verify candidates by Jaccard")
    dedup.set_defaults(func=cmd_dedup)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
