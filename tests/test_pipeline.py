import gzip
import itertools
import json
import random

import pytest

from limpia.document import Document
from limpia.io import read_jsonl, write_jsonl
from limpia.pipeline import Filter, Pipeline, Transform


class MinLength(Filter):
    def __init__(self, n: int, name: str = "min_length") -> None:
        self.n = n
        self.name = name

    def check(self, doc):
        return None if len(doc.text) >= self.n else "too_short"


class ContainsDigit(Filter):
    name = "no_digits"

    def check(self, doc):
        return "has_digit" if any(c.isdigit() for c in doc.text) else None


class Upper(Transform):
    name = "upper"

    def apply(self, doc):
        new = doc.text.upper()
        return doc.with_text(new), {
            "chars_changed": sum(a != b for a, b in zip(doc.text, new, strict=True))
        }


def _random_docs(seed: int, n: int = 300) -> list[Document]:
    rng = random.Random(seed)
    alphabet = "abcdeñó 0123"
    return [
        Document(str(i), "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))))
        for i in range(n)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_counts_are_consistent_across_steps(seed):
    pipe = Pipeline([MinLength(10), Upper(), ContainsDigit()])
    out = list(pipe.run(_random_docs(seed)))
    report = pipe.report()
    steps = report["steps"]
    for s in steps:
        assert s["kept"] + s["removed"] == s["seen"]
        assert sum(s["reasons"].values()) == s["removed"]
    for prev, nxt in itertools.pairwise(steps):
        assert nxt["seen"] == prev["kept"]
    assert steps[0]["seen"] == report["input_docs"] == 300
    assert steps[-1]["kept"] == report["output_docs"] == len(out)


def test_pipeline_matches_independent_reference():
    docs = _random_docs(7)
    pipe = Pipeline([MinLength(10), Upper(), ContainsDigit()])
    got = [(d.id, d.text) for d in pipe.run(docs)]
    expected = [
        (d.id, d.text.upper())
        for d in docs
        if len(d.text) >= 10 and not any(c.isdigit() for c in d.text)
    ]
    assert got == expected


def test_transform_counts_only_real_modifications():
    docs = [Document("a", "YA MAYÚSCULA"), Document("b", "minúscula")]
    pipe = Pipeline([Upper()])
    list(pipe.run(docs))
    stats = pipe.report()["steps"][0]
    assert stats["modified"] == 1
    assert stats["counters"] == {"chars_changed": 9}


def test_report_is_deterministic(tmp_path):
    paths = []
    for k in range(2):
        pipe = Pipeline([MinLength(10), Upper(), ContainsDigit()])
        list(pipe.run(_random_docs(3)))
        path = tmp_path / f"report{k}.json"
        pipe.write_report(path)
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_duplicate_step_names_rejected():
    with pytest.raises(ValueError):
        Pipeline([MinLength(1), MinLength(2)])


def test_documents_are_immutable():
    doc = Document("x", "hola", {"url": "u"})
    new = doc.with_text("chao").with_metadata(lang="es")
    assert doc.text == "hola" and doc.metadata == {"url": "u"}
    assert new.metadata == {"url": "u", "lang": "es"}


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_jsonl_roundtrip_preserves_unknown_fields(tmp_path, suffix):
    records = [
        {"id": "1", "text": "Señor Muñoz, ¿cómo está?", "url": "https://ejemplo.cl", "score": 0.9},
        {"id": 2, "text": "Segundo", "nested": {"a": [1, 2]}},
    ]
    src = tmp_path / f"in{suffix}"
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records) + "\n"
    if suffix.endswith(".gz"):
        with gzip.open(src, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        src.write_text(payload, encoding="utf-8")

    docs = list(read_jsonl(src))
    assert [d.id for d in docs] == ["1", "2"]
    dst = tmp_path / f"out{suffix}"
    assert write_jsonl(docs, dst) == 2
    again = list(read_jsonl(dst))
    assert again == docs
    assert again[0].metadata == {"url": "https://ejemplo.cl", "score": 0.9}


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_output_bytes_use_lf_and_are_reproducible(tmp_path, suffix):
    docs = [Document("1", "Línea uno\nlínea dos"), Document("2", "Otra")]
    first, second = tmp_path / "a" / f"out{suffix}", tmp_path / "b" / f"out{suffix}"
    write_jsonl(docs, first)
    write_jsonl(docs, second)
    # Same input, same file name: identical bytes, including the gzip header
    # (whose timestamp is fixed to 0).
    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    if suffix.endswith(".gz"):
        assert raw[4:8] == b"\x00\x00\x00\x00"  # MTIME field of the gzip header
        raw = gzip.decompress(raw)
    assert b"\r\n" not in raw
    assert raw.endswith(b"}\n")
    assert raw.count(b"\n") == 2


def test_missing_id_gets_stable_line_based_id(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")
    assert [d.id for d in read_jsonl(src)] == ["corpus:1", "corpus:3"]


@pytest.mark.parametrize(
    ("line", "message"),
    [("{not json", "invalid JSON"), ("[1, 2]", "JSON object"), ('{"id": 1}', "text")],
)
def test_malformed_lines_raise_with_line_number(tmp_path, line, message):
    src = tmp_path / "bad.jsonl"
    src.write_text('{"text": "ok"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"bad.jsonl:2: .*{message}"):
        list(read_jsonl(src))
