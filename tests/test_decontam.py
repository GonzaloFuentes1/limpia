import random

import pytest

from fixtures import CLEAN_ES, CLEAN_PT
from limpia.decontam import (
    ContaminationPolicy,
    NGramIndex,
    decontamination_step,
)
from limpia.document import Document
from limpia.pipeline import Pipeline
from limpia.text import normalize_for_matching

BENCHMARK = [
    "¿Cuál es la capital de la región de Los Ríos? Responda con el nombre de la ciudad y "
    "explique brevemente su importancia histórica para el sur del país.",
    "Lea el siguiente párrafo y determine si la conclusión del autor se sigue lógicamente "
    "de las premisas presentadas en el texto o si contiene una falacia.",
    "Pregunta corta.",
]
PASSAGE = (
    "lea el siguiente párrafo y determine si la conclusión del autor se sigue lógicamente "
    "de las premisas"
)


@pytest.fixture(scope="module")
def index():
    return NGramIndex.from_texts(BENCHMARK, n=13)


def _reference_overlap(benchmark, text, n):
    def grams(t):
        w = normalize_for_matching(t).split()
        return [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]

    bench = {g for t in benchmark for g in grams(t)}
    doc = grams(text)
    return sum(g in bench for g in doc), len(doc)


def test_index_accounting(index):
    assert index.indexed_texts == 2
    assert index.skipped_short == 1
    assert len(index) > 0


def test_planted_contamination_is_recovered_and_clean_text_untouched(index):
    planted = CLEAN_ES + " " + PASSAGE.upper().replace(" de ", ", de ") + ". Fin."
    docs = [
        Document("clean-es", CLEAN_ES),
        Document("clean-pt", CLEAN_PT),
        Document("planted", planted),
    ]
    pipe = Pipeline([decontamination_step(index, "drop")])
    kept = list(pipe.run(docs))
    assert [d.id for d in kept] == ["clean-es", "clean-pt"]
    assert kept[0] is docs[0]
    assert pipe.report()["steps"][0]["reasons"] == {"decontam:contaminated": 1}


@pytest.mark.parametrize("seed", range(10))
def test_overlap_matches_unhashed_reference(index, seed):
    rng = random.Random(seed)
    words = (CLEAN_ES + " " + BENCHMARK[0] + " " + BENCHMARK[1]).split()
    start = rng.randrange(len(words) - 40)
    text = " ".join(words[start : start + rng.randint(5, 60)])
    assert index.overlap(text) == _reference_overlap(BENCHMARK, text, 13)


def test_ngram_length_boundary():
    twelve_words = " ".join(PASSAGE.split()[:12])
    assert NGramIndex.from_texts(BENCHMARK, n=13).overlap(twelve_words)[0] == 0
    assert NGramIndex.from_texts(BENCHMARK, n=8).overlap(twelve_words) == (5, 5)


def test_flag_mode_annotates_without_dropping(index):
    docs = [Document("c", CLEAN_ES), Document("p", CLEAN_ES + " " + PASSAGE)]
    pipe = Pipeline([decontamination_step(index, "flag")])
    out = list(pipe.run(docs))
    assert [d.id for d in out] == ["c", "p"]
    assert out[0].metadata["contamination"]["flagged"] is False
    info = out[1].metadata["contamination"]
    assert info["flagged"] is True and info["matched_ngrams"] == 5  # 17 words -> 5 13-grams
    assert pipe.report()["steps"][0]["counters"] == {"flagged": 1}


def test_fraction_policy_tolerates_a_short_quote_in_a_long_document(index):
    doc = CLEAN_ES + " " + PASSAGE
    matched, total = index.overlap(doc)
    assert ContaminationPolicy().is_contaminated(matched, total)
    assert not ContaminationPolicy(min_fraction=0.2).is_contaminated(matched, total)
    assert ContaminationPolicy(min_fraction=0.2).is_contaminated(*index.overlap(PASSAGE))


def test_invalid_arguments(index):
    with pytest.raises(ValueError):
        NGramIndex(n=0)
    with pytest.raises(ValueError):
        decontamination_step(index, "delete")  # type: ignore[arg-type]
