# limpia

[![CI](https://github.com/GonzaloFuentes1/limpia/actions/workflows/ci.yml/badge.svg)](https://github.com/GonzaloFuentes1/limpia/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pretraining data curation for Spanish and Portuguese web text from Latin America. Public
pipelines such as Gopher, C4 and FineWeb were designed and calibrated on English; when they are
applied to Spanish as-is, some rules misfire for purely orthographic reasons (an English
stopword list rejects every Spanish page, the em dash that opens a line of dialogue looks like
a bullet, closing guillemets are not counted as sentence ends). `limpia` is a small, readable
toolkit that implements the standard document heuristics with per-language thresholds whose
provenance is recorded next to each value, plus exact and MinHash near-duplicate removal,
PII scrubbing for Chilean identifiers and n-gram benchmark decontamination. It is pure Python
and NumPy, streams JSONL, and every component is covered by invariant tests.

## What it is / what it is not

It is:

- A reference implementation of Gopher and FineWeb quality rules, MinHash LSH deduplication,
  regex PII scrubbing and 13-gram decontamination, with thresholds in auditable YAML.
- Deterministic: same input and seed give byte-identical outputs and reports.
- Small enough to read in an afternoon (about 1,700 lines of library code including
  docstrings) and to adapt.

It is not:

- A distributed engine. It runs in a single process; for billions of documents use
  [datatrove](https://github.com/huggingface/datatrove) or a similar framework and borrow the
  rules and thresholds from here.
- A language identifier. The stopword rule removes junk, but Portuguese text passes the
  Spanish rule and vice versa because the languages share function words. Run language
  identification upstream.
- A model-based quality classifier, a toxicity filter or a complete PII solution (names and
  street addresses are not detected).

## Quickstart

```bash
git clone https://github.com/GonzaloFuentes1/limpia && cd limpia
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev]"
source .venv/bin/activate   # .venv/Scripts/activate on Windows

# One streaming pass: quality filters, exact dedup, decontamination, PII scrubbing.
limpia run examples/sample_es.jsonl out/clean.jsonl --lang es \
    --exact-dedup --decontam examples/benchmark_es.jsonl --pii --report out/report.json
# kept 4/9 documents -> out/clean.jsonl

# Two-pass MinHash LSH near-dedup.
limpia dedup out/clean.jsonl out/dedup.jsonl --threshold 0.8 --report out/dedup_report.json
# kept 3/4 documents -> out/dedup.jsonl
```

`examples/sample_es.jsonl` contains nine short, fictional documents. In the run above the
report attributes every removal: a navigation menu and a one-line page fail the Gopher word
count, a keyword-stuffed listing fails the Gopher stopword rule (its only Spanish function
word is `en`), an upper-cased mirror of a news item is an exact duplicate, and a study guide that quotes a benchmark item is
dropped by decontamination. In the surviving contact page the email, phone and valid RUT are
replaced by `<EMAIL>`, `<PHONE>` and `<RUT>`, while the invalid RUT `12.345.678-9` is left
alone. The second command removes a lightly edited copy of the news item.

The same steps from Python:

```python
from limpia.config import load_language_config
from limpia.dedup import ExactDedupFilter
from limpia.filters import quality_filters
from limpia.io import read_jsonl, write_jsonl
from limpia.pii import PIIScrubber
from limpia.pipeline import Pipeline

pipe = Pipeline([*quality_filters(load_language_config("es")), ExactDedupFilter(), PIIScrubber()])
write_jsonl(pipe.run(read_jsonl("examples/sample_es.jsonl")), "out/clean.jsonl")
pipe.write_report("out/report.json")  # kept/removed/reasons per step
```

Input is JSONL with a `text` field (configurable with `--text-key`); every other field is
carried through unchanged. `.jsonl.gz` is read and written transparently.

## Quality filters

Thresholds live in YAML files under
[`src/limpia/resources/configs`](src/limpia/resources/configs). Every entry has a `value`, a
`source` and a one-line `rationale`; the loader refuses entries without them, unknown keys are
errors rather than silently ignored, and a `null` value disables a rule. Four configs are
bundled:

- `es` and `pt` (the default is `es`): the Gopher and FineWeb values as published for English,
  with Spanish or Portuguese stopword lists and the three orthographic adaptations below. The
  two files share their numeric values.
- `es-fineweb2` and `pt-fineweb2`: the per-language thresholds and stopword lists that
  FineWeb-2 calibrated for `spa_Latn` and `por_Latn`
  ([configs](https://github.com/huggingface/fineweb-2/tree/d0defb24f193bb9a5a11b8b14524a03c4858e1b6/configs)),
  with the rules that the FineWeb-2 pipeline disables set to `null`. For example, the Spanish
  top 2-gram limit is 0.156 instead of 0.20 and the duplicate-line limit 0.252 instead of 0.30;
  the Portuguese top 2-gram limit is 0.371. FineWeb-2 fitted these values with datatrove's
  tokenizers and also filters on a language-identification score, which limpia does not
  compute, so they transfer approximately.

The table below lists the `es` values.

| Rule | Document removed when | Value | Source |
|---|---|---|---|
| Word count | words < min or > max | 50 / 100,000 | Gopher |
| Mean word length | mean < min or > max (characters) | 3 / 10 | Gopher |
| Symbol-to-word ratio | `#` / words > max, or (`...` + `…`) / words > max | 0.1 | Gopher |
| Bullet lines | fraction of lines starting with a bullet > max | 0.9 | Gopher; bullet set adapted |
| Ellipsis lines | fraction of lines ending in an ellipsis > max | 0.3 | Gopher |
| Alphabetic words | fraction of tokens containing a letter < min | 0.8 | Gopher |
| Stopwords | distinct language stopwords present < min | 2 | Gopher rule, per-language list as in FineWeb-2 |
| Duplicate lines | fraction > max; character fraction > max | 0.30; 0.20 | Gopher |
| Duplicate paragraphs | fraction > max; character fraction > max | 0.30; 0.20 | Gopher |
| Top n-gram characters | n = 2, 3, 4: fraction > max | 0.20, 0.18, 0.16 | Gopher |
| Duplicated n-gram characters | n = 5 ... 10: fraction > max | 0.15 ... 0.10 | Gopher |
| Terminal punctuation | fraction of lines ending in it <= min | 0.12 | FineWeb; punctuation set adapted |
| Short lines | fraction of lines under 30 characters >= max | 0.67 | FineWeb |
| Duplicated line characters | fraction >= max | 0.10 | FineWeb |

Regional adaptations, each marked with `source: limpia` in the YAML:

- **Stopwords.** Hand-written Spanish and Portuguese function-word lists replace Gopher's
  English list ([`resources/stopwords`](src/limpia/resources/stopwords)).
- **Dialogue dash.** The em dash is not a bullet: Spanish and Portuguese prose open dialogue
  lines with it, and counting it would remove fiction and interviews.
- **Closing marks.** `»`, `”` and `…` count as terminal punctuation, so quoted sentences in
  Spanish typography are not treated as unterminated list items.

Comparison directions follow the papers: Gopher uses strict `>` / `<`, FineWeb uses `<=` and
`>=`. Each filter exposes `measure(text)` (raw statistics) separately from `check(doc)` (the
decision), so statistics can be logged and thresholds recalibrated on local data before they
are enforced.

## Deduplication

**Exact.** Documents are normalized (NFKC, casefold, punctuation removed, whitespace collapsed,
accents kept because `esta` and `está` are different words) and hashed to 64 bits; the first
occurrence is kept. A document whose normalized text is empty (only punctuation, symbols or
emoji) is never removed as a duplicate, by either method: normalization erased everything that
could tell two such documents apart.

**Near-duplicate.** Each document becomes a set of word 5-gram shingles (character shingles
are available). A MinHash signature of `k` values estimates Jaccard similarity, since for a
random permutation `P[min h(A) = min h(B)] = J(A, B)` (Broder, 1997); the estimate is an
average of `k` Bernoulli(J) variables. Permutations are approximated by
`h(x) = (a x + b) mod (2^31 - 1)`, chosen so that the arithmetic is exact in `uint64`.

The signature is cut into `b` bands of `r` rows. Two documents become candidates if any band
matches exactly, which happens with probability

```
P(s) = 1 - (1 - s^r)^b
```

an S-curve in the Jaccard similarity `s` that rises steeply around `(1/b)^(1/r)`. For a
target threshold `t` and a budget of `k` permutations, `limpia` picks the `(b, r)` with
`b * r <= k` that minimizes the false-positive area `integral_0^t P(s) ds` plus the
false-negative area `integral_t^1 (1 - P(s)) ds`, computed with Simpson quadrature. With
`k = 128`:

| t | (b, r) | P(t - 0.1) | P(t) | P(t + 0.1) | FP area | FN area |
|---|---|---|---|---|---|---|
| 0.6 | (18, 7) | 0.132 | 0.400 | 0.787 | 0.0334 | 0.0497 |
| 0.7 | (14, 9) | 0.132 | 0.438 | 0.867 | 0.0346 | 0.0379 |
| 0.8 | (9, 13) | 0.084 | 0.399 | 0.929 | 0.0253 | 0.0333 |

(`python benchmarks/lsh_params.py --num-perm 128`.) Weighting both error areas equally puts
`P(t)` near 0.4 rather than 0.5. Candidate pairs are merged with union-find, so clusters are
transitive, and the root of each cluster is its smallest index: the kept document is always the
first occurrence in input order. `--verify` additionally requires the estimated Jaccard of a
candidate pair to reach `t`, at a cost quadratic in bucket size.

`limpia dedup` reads the input twice: once to compute signatures (512 bytes per document at
`k = 128`) and once to write the representatives, so texts are never held in memory.

## PII scrubbing

| Type | Matched | Tag |
|---|---|---|
| Email | `user@domain.tld` | `<EMAIL>` |
| Chilean RUT | `12.345.678-5`, `12345678-5`, `...-K`; replaced only if the mod-11 check digit is valid | `<RUT>` |
| Chilean phone | `+56` or `(+56)` followed by exactly nine digits, first digit 2-9, with optional spaces, tabs, dots, dashes or parentheses; not matched when another digit follows after separators | `<PHONE>` |
| IPv4 | four octets 0-255, not part of a longer dotted number, not preceded by a version or section label (`v`, `versión`, `Art.`, `artículo`, `sección`, `N°`, ...) | `<IP>` |

Precision is favoured over recall: prices (`$9.990`), dates and population counts must not be
rewritten, which is why numbers without a valid check digit or without the country code are
left untouched. A dotted quad without a label, such as a bare section number `3.1.2.4`, is
indistinguishable from an IPv4 address and is replaced. Names and addresses are out of scope.

## Benchmark decontamination

Benchmark texts are normalized like documents and their word 13-grams are hashed into an
index (the n-gram length used for GPT-3's contamination analysis). A document is contaminated
if it has at least `--decontam-min-matches` matching n-grams and the matching fraction reaches
`--decontam-min-fraction` (defaults: one match, any fraction). `--decontam-mode flag` keeps the
document and records `{"matched_ngrams", "total_ngrams", "flagged"}` in its metadata instead of
dropping it. Benchmark items shorter than 13 words cannot be indexed; they are counted in
`NGramIndex.skipped_short` so that the caller can lower `n` for benchmarks with short items.

## How it is verified

Tests assert properties against independent references rather than re-running the
implementation:

- **MinHash estimator.** Over 150 seeds, `|J_hat - J|` stays within a Hoeffding bound
  `sqrt(log(2 T / delta) / (2 k))` (union bound over `T` trials, `delta = 1e-3`) for known
  Jaccard values; the mean is within four standard errors of `J`; the variance matches the
  binomial `J (1 - J) / k`. A naive per-permutation loop reproduces the vectorized signature.
- **LSH.** Over 400 seeds per case, the empirical candidate rate of pairs with known Jaccard
  matches `1 - (1 - s^r)^b` within binomial tolerance, for several `(b, r)`.
- **Choice of (b, r).** The quadrature equals exact rational integration of the binomial
  expansion of `P(s)` to 1e-7, and the chosen `(b, r)` equals the exact brute-force optimum.
- **Clustering.** Union-find equals BFS connected components on random graphs; pairwise
  precision and recall equal brute-force pair enumeration.
- **Filters.** Every rule has a Spanish fixture that fails with that rule's specific reason, a
  clean Spanish and a clean Portuguese text pass every rule, the repetition statistics equal
  brute-force implementations on random inputs, and the Gopher thresholds are asserted against
  the paper's values.
- **PII.** The RUT check digit agrees with an independent formulation on 2,500 random bodies;
  randomized valid RUTs are replaced while the same bodies with a wrong check digit are kept;
  phone, IP and number negatives; scrubbing is idempotent.
- **Decontamination.** Planted, reformatted benchmark passages are recovered, clean text is
  untouched, and overlap counts equal an unhashed reference.
- **Pipeline and CLI.** `kept + removed == seen` for every step and counts chain across
  steps; reports and outputs are byte-identical across runs; JSONL round-trips unknown fields.
- **Documentation.** A test re-renders the benchmark table below from
  `results/dedup_bench.json` and fails if the README differs.

CI runs `ruff check`, `ruff format --check` and `pytest` on Python 3.11 and 3.12 (CPU only).

## Deduplication benchmark

`benchmarks/dedup_bench.py` generates a seeded synthetic Spanish corpus with ground-truth
clusters and known similarities. It has two parts:

- **Near-duplicates.** Originals are random sentences over a Spanish vocabulary of about 300
  words with Zipf-like frequencies. Each original gets up to four copies with word insertions,
  deletions, substitutions and adjacent swaps at a rate drawn from {0, 1, 3, 5, 10, 20}% of
  words, plus a sentence swap with probability 0.5 when the rate is non-zero. A copy is
  labelled with its original's cluster however heavily it was edited.
- **Hard negatives.** Groups of two to four templated pages share header and footer
  boilerplate (50, 70, 85 or 95% of their words) and differ in their page-specific
  sentences. Every templated page is its own cluster, so merging two of them is a false
  positive.

For every labelled pair the true Jaccard `J` of the word 5-gram sets is computed, and the
results are reported against it. A MinHash deduplicator at threshold `t` is doing its job
when it merges pairs with `J >= t` and leaves pairs with `J < t` apart, whatever the labels
say; the templated pages show that Jaccard alone cannot tell a templated page from an edited
copy, which is where the precision loss comes from. The S-curve table checks the LSH banding
maths end to end: the measured fraction of pairs that become candidates follows
`P(s) = 1 - (1 - s^r)^b` evaluated at each pair's true Jaccard.

Throughput is the median of seven repeats on synthetic documents of about 100 words, so it
says little about web-length pages. The repeats were run on a laptop that was running other
workloads at the same time, which is why the exact-dedup repeats vary by more than a factor
of four; read the numbers as indicative only.

Generated by `benchmarks/render_table.py` from `results/dedup_bench.json`; do not edit by hand.

<!-- dedup-bench:start -->
Corpus: 6,819 documents (2,000 originals, 3,940 edited copies, 879 templated pages in 300 groups), 102 words per document on average. `J` is the true Jaccard similarity of the word 5-gram sets of a pair; a copy is merged when it ends up in its original's cluster, a template pair when its two pages end up in the same cluster.

| Method | t | (b, r) | Copies, J >= t, merged | Copies, J < t, merged | Template pairs, J >= t, merged | Template pairs, J < t, merged | Docs/s, median of 7 (range) |
|---|---|---|---|---|---|---|---|
| exact | - | - | - | - | - | - | 3,500 (3,300-14,000) |
| minhash | 0.6 | (18, 7) | 0.86 of 2,080 | 0.08 of 1,860 | 0.95 of 421 | 0.05 of 530 | 680 (550-800) |
| minhash | 0.7 | (14, 9) | 0.88 of 1,527 | 0.07 of 2,413 | 0.98 of 241 | 0.13 of 710 | 690 (550-800) |
| minhash | 0.8 | (9, 13) | 0.92 of 1,053 | 0.04 of 2,887 | 0.80 of 233 | 0.02 of 718 | 700 (560-830) |

S-curve check: fraction of labelled pairs (copies and template pairs together) that share at least one LSH band, measured, next to the prediction `mean(P(J_i))` with `P(s) = 1 - (1 - s^r)^b` over the pairs in each bin. Shown as predicted / measured.

| J | Pairs | t = 0.6, (b, r) = (18, 7) | t = 0.7, (b, r) = (14, 9) | t = 0.8, (b, r) = (9, 13) |
|---|---|---|---|---|
| 0.0-0.1 | 5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.1-0.2 | 190 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.2-0.3 | 627 | 0.002 / 0.003 | 0.000 / 0.000 | 0.000 / 0.000 |
| 0.3-0.4 | 382 | 0.013 / 0.016 | 0.001 / 0.003 | 0.000 / 0.000 |
| 0.4-0.5 | 661 | 0.084 / 0.065 | 0.015 / 0.018 | 0.001 / 0.002 |
| 0.5-0.6 | 525 | 0.250 / 0.223 | 0.069 / 0.067 | 0.005 / 0.004 |
| 0.6-0.7 | 733 | 0.621 / 0.629 | 0.287 / 0.282 | 0.043 / 0.045 |
| 0.7-0.8 | 482 | 0.902 / 0.869 | 0.638 / 0.606 | 0.193 / 0.185 |
| 0.8-0.9 | 457 | 0.998 / 0.998 | 0.963 / 0.976 | 0.666 / 0.694 |
| 0.9-1.0 | 829 | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 0.995 |

Copies merged with their original, by word edit rate:

| | 0% | 1% | 3% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| mean J | 1.00 | 0.79 | 0.68 | 0.59 | 0.41 | 0.23 |
| exact | 1.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| minhash @ 0.6 | 1.00 | 0.85 | 0.68 | 0.36 | 0.07 | 0.00 |
| minhash @ 0.7 | 1.00 | 0.71 | 0.42 | 0.16 | 0.02 | 0.00 |
| minhash @ 0.8 | 1.00 | 0.49 | 0.14 | 0.02 | 0.00 | 0.00 |

Mean J of template pairs by boilerplate share: 50%: 0.29, 70%: 0.49, 85%: 0.69, 95%: 0.85.

Pairwise scores: a pair of documents is positive if both come from the same original (7,832 true pairs, which include copy-copy pairs); precision and recall count pairs that share a predicted cluster. Every copy counts as a duplicate however heavily it was edited, so recall here mostly reflects the edit mix of the corpus.

| Method | t | Pairwise precision | Pairwise recall | Pairwise F1 |
|---|---|---|---|---|
| exact | - | 1.000 | 0.100 | 0.181 |
| minhash | 0.6 | 0.874 | 0.373 | 0.523 |
| minhash | 0.7 | 0.865 | 0.272 | 0.414 |
| minhash | 0.8 | 0.872 | 0.177 | 0.295 |

Command: `python benchmarks/dedup_bench.py --n-clusters 2000 --max-copies 4 --seed 0 --num-perm 128 --ngram 5 --n-template-groups 300 --repeats 7`. Hardware: 13th Gen Intel(R) Core(TM) i7-13620H (16 logical CPUs), Windows-11-10.0.26200-SP0, Python 3.12.13, NumPy 2.5.4. Single process; MinHash throughput includes signature computation and LSH clustering. Throughput is rounded to two significant figures; the range is the slowest and fastest of the repeats.
<!-- dedup-bench:end -->

## Design notes

- **Steps decide, the pipeline counts.** Filters return a reason or `None` and transforms
  return a new document plus counters; only the pipeline mutates statistics. This keeps the
  report consistent by construction and makes every removal attributable.
- **Shallowly immutable documents.** Transforms return new `Document` instances; fields cannot
  be reassigned and `metadata` is a read-only view of a private copy, so no step can change
  what an earlier step saw. Values nested inside `metadata` are not copied.
- **Measure, then decide.** Statistics are pure functions of text, separate from thresholds,
  which makes them testable against brute-force references and reusable for calibration.
- **Cheap rules first.** The default stack runs the O(n) Gopher rules before line rules and
  before the n-gram repetition statistics, which are the most expensive.
- **Explicit hashing.** BLAKE2b is the default; xxHash (`pip install limpia[fast]`) must be
  requested explicitly, because silently switching hash functions would change outputs
  between environments.
- **No timestamps in reports**, so two runs can be diffed.

## Project layout

```
src/limpia/
  pipeline.py        Filter / Transform contracts, Pipeline, report
  io.py, document.py streaming JSONL, read-only Document
  config.py          per-language YAML with provenance validation
  filters/           Gopher quality and repetition, FineWeb line rules
  dedup/             exact, MinHash, LSH (b, r) choice, union-find, metrics
  pii.py             email, RUT (mod 11), +56 phone, IPv4
  decontam.py        hashed n-gram index, drop / flag
  cli.py             `limpia run`, `limpia dedup`
benchmarks/          dedup benchmark, README table renderer, LSH parameter table
results/             measured benchmark output (JSON)
examples/            sample corpus and toy benchmark file
```

## References

- J. W. Rae et al. *Scaling Language Models: Methods, Analysis & Insights from Training
  Gopher.* 2021. [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)
- G. Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at
  Scale.* 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- G. Penedo et al. *FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data
  Processing to Every Language.* 2025.
  [arXiv:2506.20920](https://arxiv.org/abs/2506.20920)
- K. Lee et al. *Deduplicating Training Data Makes Language Models Better.* 2021.
  [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- A. Z. Broder. *On the resemblance and containment of documents.* Compression and Complexity
  of Sequences, 1997. [doi:10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900)
- T. B. Brown et al. *Language Models are Few-Shot Learners.* 2020 (13-gram contamination
  analysis). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- J. Leskovec, A. Rajaraman, J. D. Ullman. *Mining of Massive Datasets*, chapter 3 (LSH
  banding). [mmds.org](http://www.mmds.org/)
- [huggingface/datatrove](https://github.com/huggingface/datatrove),
  [huggingface/fineweb-2](https://github.com/huggingface/fineweb-2),
  [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup),
  [ekzhu/datasketch](https://github.com/ekzhu/datasketch) (the FP/FN-area criterion for
  choosing `(b, r)`).

`limpia` is an independent implementation of the methods described in these works.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
