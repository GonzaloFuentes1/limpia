import hashlib

import pytest

from limpia.hashing import get_hash_fn, hash64
from limpia.text import (
    normalize_for_matching,
    split_lines,
    split_paragraphs,
    split_words,
    strip_punctuation,
    word_ngrams,
)


def test_strip_punctuation_handles_spanish_marks():
    assert strip_punctuation("¿Cómo?") == "Cómo"
    assert strip_punctuation("«hola»") == "hola"
    assert strip_punctuation("¡Ya!") == "Ya"
    # Inner punctuation is part of the word.
    assert strip_punctuation("e-mail,") == "e-mail"
    assert strip_punctuation("3,5%") == "3,5"
    assert strip_punctuation("...") == ""


def test_split_words_drops_pure_punctuation_and_lowercases():
    text = "¿Dónde está   la Biblioteca?  —  Allí."
    assert split_words(text) == ["Dónde", "está", "la", "Biblioteca", "Allí"]
    assert split_words(text, lowercase=True)[3] == "biblioteca"


def test_split_lines_ignores_blank_lines():
    assert split_lines("a\n\n  \n b \r\nc") == ["a", "b", "c"]


def test_split_paragraphs():
    text = "Primer párrafo.\nSigue.\n\n  \nSegundo párrafo.\n"
    assert split_paragraphs(text) == ["Primer párrafo.\nSigue.", "Segundo párrafo."]


def test_normalize_for_matching_is_idempotent_and_keeps_accents():
    # Full-width "ABC" followed by a horizontal ellipsis.
    full_width_abc = "".join(chr(0xFF21 + k) for k in range(3))
    raw = "  ¡ESTÁ   listo!\t" + full_width_abc + "… "
    norm = normalize_for_matching(raw)
    assert norm == "está listo abc"
    assert normalize_for_matching(norm) == norm
    assert normalize_for_matching("esta") != normalize_for_matching("está")


def test_word_ngrams_counts_and_bounds():
    words = list("abcde")
    assert word_ngrams(words, 2) == [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    assert len(word_ngrams(words, 5)) == 1
    assert word_ngrams(words, 6) == []
    with pytest.raises(ValueError):
        word_ngrams(words, 0)


def test_hash64_matches_independent_blake2b_reference():
    text = "niño año pingüino"
    expected = int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"
    )
    assert hash64(text) == expected
    assert 0 <= hash64(text) < 2**64


def test_xxh64_is_explicit_and_differs_from_default():
    xxhash = pytest.importorskip("xxhash")
    assert get_hash_fn("xxh64")(b"hola") == xxhash.xxh64_intdigest(b"hola")
    assert hash64("hola", "xxh64") != hash64("hola", "blake2b")
    with pytest.raises(ValueError):
        get_hash_fn("md5")  # type: ignore[arg-type]
