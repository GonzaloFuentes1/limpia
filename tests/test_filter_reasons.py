"""One fixture per removal reason, each asserted against that exact reason.

Rules are evaluated in a fixed order and the first failing rule names the
removal, so a fixture has to pass every earlier rule and fail the target one.
The repetition fixtures are synthetic: a phrase of known length is repeated
between unique filler words, which pins the statistics inside the window
between the target threshold and the thresholds of the rules checked before
it. The measured statistics are asserted alongside the reason so that a change
of definition shows up as a numeric failure, not only as a different reason.
"""

import inspect
import random
import re

import pytest

from fixtures import CLEAN_ES
from limpia.config import load_language_config
from limpia.document import Document
from limpia.filters import (
    FineWebLineFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    fineweb,
    gopher,
    quality_filters,
)
from limpia.text import split_words

ES = load_language_config("es")
REP = ES.gopher_repetition


def _reason(f, text):
    return f.check(Document("d", text))


def _repeated_phrase(phrase: str, copies: int, gap: int) -> str:
    """``copies`` occurrences of ``phrase``, each preceded and the last followed by
    ``gap`` unique filler words, on a single line."""
    words: list[str] = []
    k = 0
    for _ in range(copies):
        for _ in range(gap):
            words.append(f"registro{k}")
            k += 1
        words.extend(phrase.split())
    for _ in range(gap):
        words.append(f"registro{k}")
        k += 1
    return " ".join(words)


# ------------------------------------------------------------- Gopher quality


def test_too_many_words_with_default_threshold():
    n = len(split_words(CLEAN_ES))
    text = "\n\n".join([CLEAN_ES] * (ES.gopher_quality.max_words // n + 1))
    f = GopherQualityFilter(ES.gopher_quality, ES.stopwords)
    assert f.measure(text)["n_words"] > ES.gopher_quality.max_words
    assert _reason(f, text) == "gopher_quality:too_many_words"


# ---------------------------------------------------------- Gopher repetition

_LONG_PHRASE = (
    "transporte escolar gratuito municipal obligatorio desde marzo para estudiantes rurales"
)

# (reason, phrase, copies, gap). Thresholds: top 2/3/4-gram 0.20/0.18/0.16,
# duplicated 5..10-gram 0.15 ... 0.10.
NGRAM_CASES = [
    ("top_2gram", "arriendo departamento", 4, 3),
    ("top_3gram", "de la municipalidad", 4, 3),
    ("top_4gram", "y de la municipalidad", 4, 5),
    ("dup_5gram", " ".join(_LONG_PHRASE.split()[:5]), 5, 14),
    ("dup_6gram", " ".join(_LONG_PHRASE.split()[:6]), 4, 15),
    ("dup_7gram", " ".join(_LONG_PHRASE.split()[:7]), 3, 14),
    ("dup_8gram", " ".join(_LONG_PHRASE.split()[:8]), 3, 17),
    ("dup_9gram", " ".join(_LONG_PHRASE.split()[:9]), 4, 27),
    ("dup_10gram", _LONG_PHRASE, 5, 36),
]


@pytest.mark.parametrize(("reason", "phrase", "copies", "gap"), NGRAM_CASES)
def test_ngram_repetition_reasons(reason, phrase, copies, gap):
    text = _repeated_phrase(phrase, copies, gap)
    f = GopherRepetitionFilter(REP)
    m = f.measure(text)
    total = len(text)
    n_words = len(phrase.split())
    if reason.startswith("top_"):
        n = int(reason[4])
        # The first n-gram of the phrase is the most frequent one (ties go to
        # the first in text order), so the statistic is known in closed form.
        gram = " ".join(phrase.split()[:n])
        assert m[f"top_{n}gram_char_frac"] == pytest.approx(copies * len(gram) / total)
        assert m[f"top_{n}gram_char_frac"] > REP.max_top_ngram_char_frac[n]
    else:
        n = int(reason[4:-4])
        assert n == n_words
        # Every later copy of the phrase is covered, and nothing else is.
        covered = (copies - 1) * len(phrase.replace(" ", ""))
        assert m[f"dup_{n}gram_char_frac"] == pytest.approx(covered / total)
        assert m[f"dup_{n}gram_char_frac"] > REP.max_dup_ngram_char_frac[n]
        if n + 1 in REP.max_dup_ngram_char_frac:
            assert m[f"dup_{n + 1}gram_char_frac"] == 0.0
    assert _reason(f, text) == f"gopher_repetition:{reason}"


_UNIQUE_LINES = [
    "El consejo escolar se reunió el martes.",
    "La reunión duró más de tres horas.",
    "Se discutió el calendario de evaluaciones.",
    "Los apoderados pidieron más talleres.",
    "La directora prometió una respuesta.",
    "El acta se publicará la próxima semana.",
    "Hubo acuerdo en extender el horario.",
    "Las clases de reforzamiento seguirán.",
    "El gimnasio será reparado en enero.",
]


def test_dup_para_frac():
    a, b, d = (
        "\n".join(_UNIQUE_LINES[0:3]),
        "\n".join(_UNIQUE_LINES[3:6]),
        "\n".join(_UNIQUE_LINES[6:9]),
    )
    c = "Fin del aviso."
    text = "\n\n".join([a, c, b, c, d, c])
    f = GopherRepetitionFilter(REP)
    m = f.measure(text)
    assert m["dup_line_frac"] == pytest.approx(2 / 12)
    assert m["dup_para_frac"] == pytest.approx(2 / 6)
    assert _reason(f, text) == "gopher_repetition:dup_para_frac"


_LONG_LINE = (
    "Recuerde que la inscripción para el taller de verano cierra el viernes a las doce horas."
)


def _dup_line_chars_text():
    lines = [_LONG_LINE, _LONG_LINE, *_UNIQUE_LINES[:3]]
    return "\n".join(lines)


def test_dup_line_char_frac_when_gopher_runs_alone():
    text = _dup_line_chars_text()
    f = GopherRepetitionFilter(REP)
    m = f.measure(text)
    assert m["dup_line_frac"] == pytest.approx(1 / 5)
    assert m["dup_line_char_frac"] == pytest.approx(len(_LONG_LINE) / len(text))
    assert m["dup_line_char_frac"] > REP.max_dup_line_char_frac
    assert _reason(f, text) == "gopher_repetition:dup_line_char_frac"


def test_gopher_dup_line_char_rule_is_subsumed_by_fineweb_in_the_default_stack():
    # The same duplicated-line characters over a denominator that is never
    # smaller, with a higher threshold: whenever Gopher's rule fires, FineWeb's
    # has already fired (it runs earlier in the default stack).
    text = _dup_line_chars_text()
    assert _reason(FineWebLineFilter(ES.fineweb), text) == "fineweb_lines:dup_line_char_frac"
    rng = random.Random(0)
    gopher_rep, fineweb_lines = GopherRepetitionFilter(REP), FineWebLineFilter(ES.fineweb)
    order = [f.name for f in quality_filters(ES)]
    assert order.index("fineweb_lines") < order.index("gopher_repetition")
    for _ in range(300):
        lines = [rng.choice([*_UNIQUE_LINES, _LONG_LINE]) for _ in range(rng.randint(1, 12))]
        text = "\n".join(lines)
        g = gopher_rep.measure(text)["dup_line_char_frac"]
        fw = fineweb_lines.measure(text)["dup_line_char_frac"]
        assert g <= fw
        if g > REP.max_dup_line_char_frac:
            assert fw >= ES.fineweb.max_dup_line_char_frac


def test_dup_para_char_frac():
    # A repeated stanza whose continuation lines are indented: the indentation
    # counts in the paragraph characters but not in the stripped lines, so the
    # paragraph rule fires while the line rules stay under their thresholds.
    indent = "\n" + " " * 8
    stanza = indent.join(
        ["El río baja lento", "entre piedras grises", "y la tarde cae", "sobre los tejados"]
    )
    u1 = "\n".join(_UNIQUE_LINES[0:2])
    u2 = "\n".join(_UNIQUE_LINES[2:4])
    u3 = "\n".join(_UNIQUE_LINES[4:6])
    text = "\n\n".join([u1, stanza, u2, u3, stanza])
    f = GopherRepetitionFilter(REP)
    m = f.measure(text)
    assert m["dup_line_frac"] == pytest.approx(4 / 14)
    assert m["dup_para_frac"] == pytest.approx(1 / 5)
    assert m["dup_line_char_frac"] <= REP.max_dup_line_char_frac
    assert m["dup_para_char_frac"] == pytest.approx(len(stanza) / len(text))
    assert m["dup_para_char_frac"] > REP.max_dup_para_char_frac
    assert _reason(f, text) == "gopher_repetition:dup_para_char_frac"


def test_every_reason_has_a_fixture():
    """Guard: the reasons the filters can return are all exercised above or in
    test_quality_filters.py."""
    covered = {
        # test_quality_filters.py
        "gopher_quality:too_few_words",
        "gopher_quality:short_mean_word_length",
        "gopher_quality:long_mean_word_length",
        "gopher_quality:hash_ratio",
        "gopher_quality:ellipsis_ratio",
        "gopher_quality:bullet_lines",
        "gopher_quality:ellipsis_lines",
        "gopher_quality:alpha_words_ratio",
        "gopher_quality:too_few_stopwords",
        "gopher_repetition:dup_line_frac",
        "fineweb_lines:empty",
        "fineweb_lines:terminal_punct_ratio",
        "fineweb_lines:short_line_ratio",
        "fineweb_lines:dup_line_char_frac",
        # this module
        "gopher_quality:too_many_words",
        "gopher_repetition:dup_para_frac",
        "gopher_repetition:dup_line_char_frac",
        "gopher_repetition:dup_para_char_frac",
        *(f"gopher_repetition:{reason}" for reason, *_ in NGRAM_CASES),
    }
    source = inspect.getsource(gopher) + inspect.getsource(fineweb)
    literal = set(re.findall(r'"((?:gopher_quality|gopher_repetition|fineweb_lines):\w+)"', source))
    templated = {f"gopher_repetition:top_{n}gram" for n in REP.max_top_ngram_char_frac} | {
        f"gopher_repetition:dup_{n}gram" for n in REP.max_dup_ngram_char_frac
    }
    assert literal | templated == covered
