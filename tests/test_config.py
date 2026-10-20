import copy
import re

import pytest
import yaml

from limpia.config import (
    available_languages,
    load_language_config,
    load_stopwords,
    parse_language_config,
)

PAPERS = {"2112.11446", "2406.17557", "2506.20920"}


def _raw(lang: str = "es") -> dict:
    from importlib import resources

    path = resources.files("limpia") / "resources" / "configs" / f"{lang}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


BUNDLED = ["es", "es-fineweb2", "pt", "pt-fineweb2"]


def test_bundled_languages():
    assert available_languages() == BUNDLED


@pytest.mark.parametrize("lang", BUNDLED)
def test_every_threshold_has_value_source_and_rationale(lang):
    cfg = load_language_config(lang)
    assert cfg.language == lang.split("-")[0]
    n_entries = 0
    for section in cfg.provenance.values():
        for entry in section.values():
            n_entries += 1
            assert entry["source"] in cfg.sources
            assert len(entry["rationale"]) > 20
    assert n_entries == 22
    arxiv_ids = {
        m.group(1)
        for src in cfg.sources.values()
        if (m := re.fullmatch(r"https://arxiv\.org/abs/(\d{4}\.\d{5})", src["url"]))
    }
    assert arxiv_ids == PAPERS


def test_gopher_thresholds_match_the_paper():
    cfg = load_language_config("es")
    q, r = cfg.gopher_quality, cfg.gopher_repetition
    assert (q.min_words, q.max_words) == (50, 100_000)
    assert (q.min_mean_word_length, q.max_mean_word_length) == (3.0, 10.0)
    assert (q.max_symbol_word_ratio, q.max_bullet_lines_ratio) == (0.1, 0.9)
    assert (q.max_ellipsis_lines_ratio, q.min_alpha_words_ratio) == (0.3, 0.8)
    assert r.max_top_ngram_char_frac == {2: 0.20, 3: 0.18, 4: 0.16}
    assert r.max_dup_ngram_char_frac == {n: round(0.15 - 0.01 * (n - 5), 2) for n in range(5, 11)}
    assert "—" not in q.bullet_prefixes
    # Rules that the paper-based configs leave out are explicitly disabled.
    assert cfg.fineweb.max_newline_word_ratio is None


# Values copied from huggingface/fineweb-2 at commit d0defb2 (configs/spa_Latn.yml,
# configs/por_Latn.yml and fineweb-2-pipeline.py).
FINEWEB2 = {
    "es-fineweb2": {
        "max_mean_word_length": 14.0,
        "min_alpha_words_ratio": 0.825,
        "max_dup_line_frac": 0.252,
        "top": {2: 0.156, 3: 0.14, 4: 0.125},
        "dup": {5: 0.145, 6: 0.136, 7: 0.127, 8: 0.117, 9: 0.108, 10: 0.099},
        "line_punct": 0.182,
        "new_line_ratio": 0.138,
        "n_stopwords": 16,
    },
    "pt-fineweb2": {
        "max_mean_word_length": 13.0,
        "min_alpha_words_ratio": 0.814,
        "max_dup_line_frac": 0.287,
        "top": {2: 0.371, 3: 0.191, 4: 0.163},
        "dup": {5: 0.163, 6: 0.153, 7: 0.141, 8: 0.13, 9: 0.119, 10: 0.108},
        "line_punct": 0.077,
        "new_line_ratio": 0.186,
        "n_stopwords": 16,
    },
}


@pytest.mark.parametrize("lang", sorted(FINEWEB2))
def test_fineweb2_configs_match_the_published_values(lang):
    ref = FINEWEB2[lang]
    cfg = load_language_config(lang)
    q, r, f = cfg.gopher_quality, cfg.gopher_repetition, cfg.fineweb
    assert q.max_mean_word_length == ref["max_mean_word_length"]
    assert q.min_alpha_words_ratio == ref["min_alpha_words_ratio"]
    assert (q.min_words, q.min_mean_word_length, q.min_stopwords) == (50, 3.0, 2)
    assert r.max_dup_line_frac == ref["max_dup_line_frac"]
    assert r.max_top_ngram_char_frac == ref["top"]
    assert r.max_dup_ngram_char_frac == ref["dup"]
    # Disabled in the FineWeb-2 pipeline.
    assert r.max_dup_para_frac is None
    assert r.max_dup_line_char_frac is None
    assert r.max_dup_para_char_frac is None
    assert f.max_short_line_ratio is None
    assert f.min_terminal_punct_ratio == ref["line_punct"]
    assert f.max_newline_word_ratio == ref["new_line_ratio"]
    assert f.max_dup_line_char_frac == 0.1
    assert len(cfg.stopwords) == ref["n_stopwords"]


def test_stopword_lists_are_clean_and_language_specific():
    es, pt = load_stopwords("es"), load_stopwords("pt")
    assert {"de", "que", "el", "los"} <= es
    assert {"de", "que", "não", "você"} <= pt
    assert not any(w.startswith("#") or w != w.strip().lower() for w in es | pt)
    assert "não" not in es and "los" not in pt


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["gopher_quality"]["min_words"].pop("rationale"), "rationale"),
        (lambda r: r["gopher_quality"].pop("min_words"), "missing keys"),
        (lambda r: r["fineweb"].update({"typo_key": r["fineweb"]["short_line_chars"]}), "unknown"),
        (lambda r: r["fineweb"]["short_line_chars"].update({"source": "blog"}), "unknown source"),
        (lambda r: r["gopher_quality"]["min_words"].update({"value": 50.5}), "integer"),
        (lambda r: r["sources"]["gopher"].pop("url"), "citation"),
        (lambda r: r["gopher_quality"]["min_words"].update({"value": None}), "null"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_language_config(raw)


def test_load_from_path(tmp_path):
    path = tmp_path / "custom.yaml"
    raw = _raw()
    raw["gopher_quality"]["min_words"]["value"] = 20
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_language_config(path).gopher_quality.min_words == 20
    with pytest.raises(ValueError, match="available"):
        load_language_config("xx")
