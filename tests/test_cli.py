"""End-to-end checks of the CLI on the bundled example corpus."""

import json
from pathlib import Path

import pytest

from limpia.cli import main
from limpia.io import read_jsonl

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SAMPLE = EXAMPLES / "sample_es.jsonl"
BENCH = EXAMPLES / "benchmark_es.jsonl"


def _run(tmp_path: Path, *extra: str) -> tuple[list, dict]:
    out, report = tmp_path / "clean.jsonl", tmp_path / "report.json"
    code = main(["run", str(SAMPLE), str(out), "--report", str(report), *extra])
    assert code == 0
    return list(read_jsonl(out)), json.loads(report.read_text(encoding="utf-8"))


def test_full_run_on_sample(tmp_path):
    docs, report = _run(
        tmp_path, "--exact-dedup", "--pii", "--decontam", str(BENCH), "--lang", "es"
    )
    assert [d.id for d in docs] == ["noticia-1", "noticia-1-editada", "receta", "contacto"]
    reasons = {r: n for s in report["steps"] for r, n in s["reasons"].items()}
    assert reasons == {
        "gopher_quality:too_few_words": 2,
        "gopher_quality:too_few_stopwords": 1,
        "exact_dedup:duplicate": 1,
        "decontam:contaminated": 1,
    }
    contacto = docs[-1]
    assert "<EMAIL>" in contacto.text and "<PHONE>" in contacto.text and "<RUT>" in contacto.text
    assert "12.345.678-9" in contacto.text  # invalid RUT left untouched
    assert contacto.metadata == {"url": "https://ejemplo.cl/subsidio"}


def test_run_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _run(a, "--exact-dedup", "--pii")
    _run(b, "--exact-dedup", "--pii")
    for name in ("clean.jsonl", "report.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_flag_mode_keeps_contaminated_document(tmp_path):
    docs, _ = _run(tmp_path, "--no-quality", "--decontam", str(BENCH), "--decontam-mode", "flag")
    flagged = [d.id for d in docs if d.metadata["contamination"]["flagged"]]
    assert flagged == ["contaminada"]


def test_dedup_command_removes_near_duplicate(tmp_path):
    clean, _ = _run(tmp_path, "--exact-dedup")
    out, report = tmp_path / "dedup.jsonl", tmp_path / "dedup.json"
    code = main(["dedup", str(tmp_path / "clean.jsonl"), str(out), "--report", str(report)])
    assert code == 0
    kept = [d.id for d in read_jsonl(out)]
    assert "noticia-1-editada" not in kept and "noticia-1" in kept
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["input_docs"] == len(clean)
    assert payload["removed"] == len(clean) - len(kept) == 1
    assert payload["config"]["bands"] * payload["config"]["rows"] <= 128


def test_report_goes_to_stdout_by_default(tmp_path, capsys):
    assert main(["run", str(SAMPLE), str(tmp_path / "o.jsonl"), "--no-quality", "--pii"]) == 0
    assert json.loads(capsys.readouterr().out)["input_docs"] == 9


def test_no_steps_is_an_error(tmp_path):
    assert main(["run", str(SAMPLE), str(tmp_path / "o.jsonl"), "--no-quality"]) == 2
    with pytest.raises(SystemExit):
        main(["run", str(SAMPLE)])
