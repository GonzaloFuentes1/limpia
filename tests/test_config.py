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


def test_bundled_languages():
    assert available_languages() == ["es", "pt"]


@pytest.mark.parametrize("lang", ["es", "pt"])
def test_every_threshold_has_value_source_and_rationale(lang):
    cfg = load_language_config(lang)
    assert cfg.language == lang
    n_entries = 0
    for section in cfg.provenance.values():
        for entry in section.values():
            n_entries += 1
            assert entry["source"] in cfg.sources
            assert len(entry["rationale"]) > 20
    assert n_entries == 21
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
