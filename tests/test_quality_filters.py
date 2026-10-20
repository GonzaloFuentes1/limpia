import dataclasses
import random

import pytest

from fixtures import CLEAN_ES, CLEAN_PT, ENGLISH
from limpia.config import load_language_config
from limpia.document import Document
from limpia.filters import (
    FineWebLineFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    quality_filters,
)
from limpia.filters.repetition import (
    duplicate_fractions,
    duplicated_ngram_char_fraction,
    top_ngram_char_fraction,
)
from limpia.pipeline import Pipeline

ES = load_language_config("es")
PT = load_language_config("pt")


def _gq(cfg=ES, **overrides):
    return GopherQualityFilter(dataclasses.replace(cfg.gopher_quality, **overrides), cfg.stopwords)


def _check(f, text):
    return f.check(Document("d", text))


# --------------------------------------------------------------------------- clean text


@pytest.mark.parametrize(("cfg", "text"), [(ES, CLEAN_ES), (PT, CLEAN_PT)])
def test_clean_text_passes_every_default_filter(cfg, text):
    for f in quality_filters(cfg):
        assert _check(f, text) is None, f.name


# ------------------------------------------------------------------- Gopher quality


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Hola a todos, ¿cómo están?", "too_few_words"),
        (" ".join(["a y e o"] * 15), "short_mean_word_length"),
        (" ".join(["responsabilidades internacionalización"] * 30), "long_mean_word_length"),
        (CLEAN_ES + " " + " ".join(f"#tendencia{i}" for i in range(30)), "symbol_word_ratio"),
        (
            "\n".join(f"• Oferta {i} de zapatillas para correr con envío gratis" for i in range(9)),
            "bullet_lines",
        ),
        (
            "\n".join(
                f"Lea la nota completa número {i} sobre la elección regional de este año..."
                for i in range(8)
            ),
            "ellipsis_lines",
        ),
        (CLEAN_ES + " " + " ".join(str(1000 + i) for i in range(60)), "alpha_words_ratio"),
        (ENGLISH, "too_few_stopwords"),
    ],
)
def test_gopher_quality_rejects_with_specific_reason(text, reason):
    assert _check(_gq(), text) == f"gopher_quality:{reason}"


def test_gopher_quality_max_words_bound():
    assert _check(_gq(max_words=60), CLEAN_ES) == "gopher_quality:too_many_words"


def test_spanish_dialogue_dash_is_not_a_bullet():
    dialogue = "\n".join(
        f"—¿Quién llamó a la puerta número {i}? —preguntó la abuela." for i in range(12)
    )
    bullets = dialogue.replace("—¿", "• ¿")
    f = _gq()
    assert f.measure(dialogue)["bullet_lines_ratio"] == 0.0
    assert f.measure(bullets)["bullet_lines_ratio"] == 1.0


def test_stopword_requirement_is_language_specific():
    # Spanish text against the Portuguese list still has shared function words
    # (de, que, para); the rule is a junk detector, not a language identifier.
    assert _check(_gq(PT), CLEAN_ES) is None
    m = _gq().measure(CLEAN_ES)
    assert m["stopwords"] >= 30
    assert _gq().measure(ENGLISH)["stopwords"] == 0


def test_gopher_quality_measure_values():
    text = "uno dos tres\n• cuatro...\ncinco 123"
    m = _gq().measure(text)
    assert m["n_words"] == 6
    assert m["mean_word_length"] == pytest.approx(len("unodostrescuatrocinco123") / 6)
    assert m["symbol_word_ratio"] == pytest.approx(1 / 6)
    assert m["bullet_lines_ratio"] == pytest.approx(1 / 3)
    assert m["ellipsis_lines_ratio"] == pytest.approx(1 / 3)
    # Tokens: uno dos tres • cuatro... cinco 123 -> the bullet and the number lack letters.
    assert m["alpha_words_ratio"] == pytest.approx(5 / 7)


# -------------------------------------------------------- repetition: parity tests


def _ref_duplicated_ngram_char_fraction(words, n):
    """Brute force: a word is covered if some n-gram containing it occurred earlier."""
    total = sum(len(w) for w in words)
    if total == 0 or len(words) < n:
        return 0.0
    covered = 0
    for i, w in enumerate(words):
        starts = range(max(0, i - n + 1), min(i, len(words) - n) + 1)
        if any(any(words[j : j + n] == words[k : k + n] for j in range(k)) for k in starts):
            covered += len(w)
    return covered / total


def _ref_top_ngram_char_fraction(words, n):
    total = sum(len(w) for w in words)
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    best = 0
    for g in set(grams):
        c = grams.count(g)
        if c > 1:
            best = max(best, c * sum(map(len, g)))
    return min(best / total, 1.0) if total else 0.0


@pytest.mark.parametrize("seed", range(25))
def test_repetition_statistics_match_brute_force(seed):
    rng = random.Random(seed)
    vocab = ["el", "perro", "come", "pan", "y", "agua", "fría", "sol"][: rng.randint(2, 8)]
    words = [rng.choice(vocab) for _ in range(rng.randint(0, 40))]
    for n in range(1, 8):
        assert duplicated_ngram_char_fraction(words, n) == pytest.approx(
            _ref_duplicated_ngram_char_fraction(words, n)
        )
        assert top_ngram_char_fraction(words, n) == pytest.approx(
            _ref_top_ngram_char_fraction(words, n)
        )
        assert 0.0 <= duplicated_ngram_char_fraction(words, n) <= 1.0


def test_duplicate_fractions_counts_only_repeats():
    frac, char_frac = duplicate_fractions(["abc", "de", "abc", "abc", "f"])
    assert frac == pytest.approx(2 / 5)
    assert char_frac == pytest.approx(6 / 12)
    assert duplicate_fractions([]) == (0.0, 0.0)


# ------------------------------------------------------------- Gopher repetition


def test_repeated_boilerplate_lines_are_rejected():
    text = CLEAN_ES + "\n" + "\n".join(["Suscríbete a nuestro boletín semanal."] * 10)
    assert _check(GopherRepetitionFilter(ES.gopher_repetition), text) == (
        "gopher_repetition:dup_line_frac"
    )


def test_keyword_stuffing_is_rejected_by_top_ngram():
    stuffed = CLEAN_ES + " " + " ".join(["arriendo departamento"] * 40)
    f = GopherRepetitionFilter(ES.gopher_repetition)
    assert f.measure(stuffed)["top_2gram_char_frac"] > 0.2
    assert _check(f, stuffed) == "gopher_repetition:top_2gram"


def test_spun_text_is_rejected_by_duplicated_ngrams():
    paragraph = CLEAN_ES.split("\n")[0]
    spun = paragraph + " " + paragraph.replace("invierno", "verano")
    f = GopherRepetitionFilter(ES.gopher_repetition)
    m = f.measure(spun)
    assert m["dup_10gram_char_frac"] > 0.3
    assert _check(f, spun).startswith("gopher_repetition:")


def test_paragraph_statistics():
    text = "Primer párrafo.\n\nSegundo párrafo largo.\n\nPrimer párrafo."
    m = GopherRepetitionFilter(ES.gopher_repetition).measure(text)
    assert m["dup_para_frac"] == pytest.approx(1 / 3)
    assert m["dup_para_char_frac"] == pytest.approx(15 / (15 + 22 + 15))


# --------------------------------------------------------------------- FineWeb


def test_menu_without_punctuation_is_rejected():
    menu = "\n".join(["Inicio", "Noticias", "Deportes", "Contacto", "Quiénes somos"] * 3)
    assert _check(FineWebLineFilter(ES.fineweb), menu) == "fineweb_lines:terminal_punct_ratio"


def test_short_lines_are_rejected():
    text = CLEAN_ES + "\n" + "\n".join(f"Sí, {i}." for i in range(20))
    assert _check(FineWebLineFilter(ES.fineweb), text) == "fineweb_lines:short_line_ratio"


def test_duplicated_line_characters_are_rejected():
    first_line = CLEAN_ES.split("\n")[0]
    text = CLEAN_ES + "\n" + first_line
    f = FineWebLineFilter(ES.fineweb)
    assert f.measure(text)["dup_line_char_frac"] >= 0.1
    assert _check(f, text) == "fineweb_lines:dup_line_char_frac"


def test_spanish_closing_quotes_count_as_terminal():
    text = "Dijo: «vamos a la playa»\nY ella respondió: “mañana”"
    assert FineWebLineFilter(ES.fineweb).measure(text)["terminal_punct_ratio"] == 1.0


def test_fineweb_comparisons_are_inclusive_as_published():
    cfg = dataclasses.replace(ES.fineweb, min_terminal_punct_ratio=0.5)
    two_lines = (
        "Una línea larga que termina con un punto final.\nOtra línea larga que no termina así"
    )
    assert _check(FineWebLineFilter(cfg), two_lines) == "fineweb_lines:terminal_punct_ratio"
    assert _check(FineWebLineFilter(ES.fineweb), "") == "fineweb_lines:empty"


def test_default_stack_in_a_pipeline_reports_reasons():
    docs = [
        Document("ok", CLEAN_ES),
        Document("short", "Hola."),
        Document("menu", "\n".join(["Inicio", "Noticias", "Contacto"] * 20)),
        Document("en", ENGLISH),
    ]
    pipe = Pipeline(quality_filters(ES))
    assert [d.id for d in pipe.run(docs)] == ["ok"]
    reasons = {r for s in pipe.report()["steps"] for r in s["reasons"]}
    # The menu has enough words but no Spanish function words, so the cheap
    # stopword rule removes it before the line-level rules are evaluated.
    assert reasons == {"gopher_quality:too_few_words", "gopher_quality:too_few_stopwords"}
