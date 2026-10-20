import dataclasses
import random

import pytest

from fixtures import CLEAN_ES, CLEAN_PT, ENGLISH, ENGLISH_SHARED_STOPWORDS
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
        (CLEAN_ES + " " + " ".join(f"#tendencia{i}" for i in range(30)), "hash_ratio"),
        (CLEAN_ES + " " + " ".join(f"y así{i}…" for i in range(30)), "ellipsis_ratio"),
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
    assert m["distinct_stopwords"] >= 20
    assert _gq().measure(ENGLISH)["distinct_stopwords"] == 1


def test_stopword_rule_counts_distinct_words():
    # Gopher asks for "at least two of the following words": repeating one
    # stopword many times does not satisfy it.
    junk = " ".join(f"zapatilla{i} negra a talla grande" for i in range(12))
    assert _gq().measure(junk)["distinct_stopwords"] == 1
    assert _check(_gq(), junk) == "gopher_quality:too_few_stopwords"


def test_stopword_rule_is_not_a_language_identifier():
    assert _gq().measure(ENGLISH_SHARED_STOPWORDS)["distinct_stopwords"] == 2
    for f in quality_filters(ES):
        assert _check(f, ENGLISH_SHARED_STOPWORDS) is None, f.name


def test_hash_and_ellipsis_ratios_are_separate():
    # 100 words with 6 hashes and 6 ellipses: each ratio is 0.06, below 0.1,
    # so Gopher keeps the document although the combined count is 0.12.
    words = [f"palabra{i}" for i in range(100)]
    for i in range(6):
        words[i] = "#" + words[i]
        words[50 + i] += "…"
    m = _gq().measure(" ".join(words))
    assert m["hash_word_ratio"] == pytest.approx(0.06)
    assert m["ellipsis_word_ratio"] == pytest.approx(0.06)
    q = dataclasses.replace(ES.gopher_quality, min_stopwords=0)
    assert _check(GopherQualityFilter(q, ES.stopwords), " ".join(words)) is None


def test_gopher_quality_measure_values():
    text = "uno dos tres\n• cuatro...\ncinco 123"
    m = _gq().measure(text)
    assert m["n_words"] == 6
    assert m["mean_word_length"] == pytest.approx(len("unodostrescuatrocinco123") / 6)
    assert m["hash_word_ratio"] == 0.0
    assert m["ellipsis_word_ratio"] == pytest.approx(1 / 6)
    assert m["bullet_lines_ratio"] == pytest.approx(1 / 3)
    assert m["ellipsis_lines_ratio"] == pytest.approx(1 / 3)
    # Tokens: uno dos tres • cuatro... cinco 123 -> the bullet and the number lack letters.
    assert m["alpha_words_ratio"] == pytest.approx(5 / 7)


# -------------------------------------------------------- repetition: parity tests


def _ref_duplicated_ngram_char_fraction(words, n, total):
    """Brute force: a word is covered if some n-gram containing it occurred earlier."""
    if total == 0 or len(words) < n:
        return 0.0
    covered = 0
    for i, w in enumerate(words):
        starts = range(max(0, i - n + 1), min(i, len(words) - n) + 1)
        if any(any(words[j : j + n] == words[k : k + n] for j in range(k)) for k in starts):
            covered += len(w)
    return covered / total


def _ref_top_ngram_char_fraction(words, n, total):
    """Written from Gopher's sentence, not from the implementation.

    "The fraction of characters contained within the most frequently-occurring
    n-gram" (Rae et al., 2021, Appendix A): find the highest frequency, take
    the first n-gram (in text order) with that frequency, and count the
    characters of all its occurrences. A frequency of one is no repetition.
    """
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams or total == 0:
        return 0.0
    top = max(grams.count(g) for g in grams)
    if top < 2:
        return 0.0
    first = next(g for g in grams if grams.count(g) == top)
    return min(top * len(first) / total, 1.0)


@pytest.mark.parametrize("seed", range(25))
def test_repetition_statistics_match_brute_force(seed):
    rng = random.Random(seed)
    vocab = ["el", "perro", "come", "pan", "y", "agua", "fría", "sol"][: rng.randint(2, 8)]
    words = [rng.choice(vocab) for _ in range(rng.randint(0, 40))]
    total = len(" ".join(words))
    for n in range(1, 8):
        dup = duplicated_ngram_char_fraction(words, n, total_chars=total)
        assert dup == pytest.approx(_ref_duplicated_ngram_char_fraction(words, n, total))
        assert top_ngram_char_fraction(words, n, total_chars=total) == pytest.approx(
            _ref_top_ngram_char_fraction(words, n, total)
        )
        assert 0.0 <= dup <= 1.0


def test_top_ngram_uses_the_most_frequent_ngram_not_the_largest_coverage():
    # "x y" occurs three times (3 * 3 = 9 characters); "universidad nacional"
    # occurs only twice but covers 2 * 20 = 40 characters. Gopher scores the
    # most frequent n-gram, so the answer is 9 / len(text).
    text = "x y x y x y la universidad nacional la universidad nacional"
    words = text.split()
    assert top_ngram_char_fraction(words, 2, total_chars=len(text)) == pytest.approx(9 / len(text))
    # Ties go to the first n-gram in text order: "ab" before "cd".
    tie = "ab cd ab cd ef"
    assert top_ngram_char_fraction(tie.split(), 1, total_chars=len(tie)) == pytest.approx(4 / 14)
    assert top_ngram_char_fraction(tie.split(), 2, total_chars=len(tie)) == pytest.approx(10 / 14)
    assert top_ngram_char_fraction(["a", "b", "c"], 2, total_chars=5) == 0.0


def test_character_fractions_divide_by_document_length():
    # len(text) = 7 + 1 + 7 + 1 + 4 = 20, including the two newlines.
    text = "uno dos\nuno dos\ntres"
    m = GopherRepetitionFilter(ES.gopher_repetition).measure(text)
    assert m["dup_line_frac"] == pytest.approx(1 / 3)
    assert m["dup_line_char_frac"] == pytest.approx(7 / 20)
    # "uno dos" twice: 2 * 7 characters.
    assert m["top_2gram_char_frac"] == pytest.approx(14 / 20)
    # FineWeb divides by the characters without newlines, as datatrove does.
    assert FineWebLineFilter(ES.fineweb).measure(text)["dup_line_char_frac"] == pytest.approx(
        7 / 18
    )


# A natural paragraph that repeats one long phrase three times. Gopher's
# statistic gives top 4-gram 3 * len("el programa de reciclaje") / len(text),
# well under 0.16; scoring the n-gram with the largest coverage instead picks
# "programa de reciclaje domiciliario" and rejects the paragraph.
MUNICIPAL_NEWS = (
    "La municipalidad de Valle Hermoso informó este lunes que el programa de reciclaje "
    "domiciliario llegará a todos los barrios de la comuna antes de fin de año. Según el "
    "alcalde, el programa de reciclaje domiciliario ya recibe más de cuatro toneladas de "
    "residuos cada semana. Los vecinos pueden inscribirse en la oficina de medio ambiente o en "
    "el sitio web de la municipalidad. La encargada del proyecto explicó que el programa de "
    "reciclaje domiciliario funciona con camiones propios y con recicladores de base que "
    "reciben un pago mensual. La próxima etapa incluirá puntos limpios en las escuelas y en la "
    "plaza central de la comuna."
)


def test_natural_paragraph_with_a_repeated_phrase_passes():
    f = GopherRepetitionFilter(ES.gopher_repetition)
    m = f.measure(MUNICIPAL_NEWS)
    total = len(MUNICIPAL_NEWS)
    assert m["top_4gram_char_frac"] == pytest.approx(3 * len("el programa de reciclaje") / total)
    assert m["top_4gram_char_frac"] < 0.16
    largest_coverage = 3 * len("programa de reciclaje domiciliario") / total
    assert largest_coverage > 0.16
    for step in quality_filters(ES):
        assert _check(step, MUNICIPAL_NEWS) is None, step.name


def test_duplicate_fractions_counts_only_repeats():
    frac, char_frac = duplicate_fractions(["abc", "de", "abc", "abc", "f"], total_chars=16)
    assert frac == pytest.approx(2 / 5)
    assert char_frac == pytest.approx(6 / 16)
    assert duplicate_fractions([], total_chars=0) == (0.0, 0.0)


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
    assert m["dup_para_char_frac"] == pytest.approx(15 / len(text))


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
