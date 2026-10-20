import itertools
import random

import pytest

from limpia.document import Document
from limpia.pii import PIIScrubber, is_valid_rut, rut_check_digit, scrub
from limpia.pipeline import Pipeline


def _reference_check_digit(body: int) -> str:
    """Independent formulation: weights 2..7 cycled over the reversed digit string."""
    weights = itertools.cycle(range(2, 8))
    s = sum(int(d) * w for d, w in zip(reversed(str(body)), weights, strict=False))
    dv = (11 - s % 11) % 11
    return "K" if dv == 10 else str(dv)


def _fmt(body: int, dv: str, dotted: bool) -> str:
    if dotted:
        return f"{body:,}".replace(",", ".") + "-" + dv
    return f"{body}-{dv}"


@pytest.mark.parametrize("seed", range(5))
def test_check_digit_matches_independent_reference(seed):
    rng = random.Random(seed)
    for _ in range(500):
        body = rng.randint(1_000_000, 99_999_999)
        assert rut_check_digit(body) == _reference_check_digit(body)


def test_known_ruts():
    assert is_valid_rut("11.111.111-1")
    assert is_valid_rut("12.345.678-5")
    assert is_valid_rut("12345678-5")
    assert not is_valid_rut("12.345.678-9")
    assert not is_valid_rut("12.345.678")
    k_body = next(b for b in range(10_000_000, 10_000_100) if rut_check_digit(b) == "K")
    assert is_valid_rut(f"{k_body}-k") and is_valid_rut(f"{k_body}-K")


@pytest.mark.parametrize("seed", range(5))
def test_valid_ruts_are_replaced_and_invalid_ones_kept(seed):
    rng = random.Random(seed)
    for _ in range(100):
        body = rng.randint(1_000_000, 99_999_999)
        dv = rut_check_digit(body)
        dotted = rng.random() < 0.5
        good = _fmt(body, dv.lower() if rng.random() < 0.5 else dv, dotted)
        wrong_dv = rng.choice([d for d in "0123456789K" if d != dv])
        bad = _fmt(body, wrong_dv, dotted)
        text = f"El contribuyente con RUT {good} declaró; el folio {bad} no corresponde."
        out, counts = scrub(text)
        assert out == f"El contribuyente con RUT <RUT> declaró; el folio {bad} no corresponde."
        assert counts["RUT"] == 1


@pytest.mark.parametrize(
    "phone",
    [
        "+56 9 8765 4321",
        "+56987654321",
        "+56 2 2345 6789",
        "+56-32-123-4567",
        "(+56) 9 1234 5678",
        "+56 (2) 2345 6789",
    ],
)
def test_chilean_phone_formats(phone):
    out, counts = scrub(f"Llámenos al {phone} de lunes a viernes.")
    assert out == "Llámenos al <PHONE> de lunes a viernes."
    assert counts["PHONE"] == 1


@pytest.mark.parametrize(
    "not_phone",
    [
        "+56 1 2345 6789",  # Chilean numbers never start with 1
        "+56 9 1234 567",  # eight digits
        "+56 9 1234 56789",  # ten digits
        "+1 415 555 0100",  # other country
        "9 8765 4321",  # no country code: too ambiguous with other numbers
    ],
)
def test_phone_negatives(not_phone):
    text = f"Número: {not_phone}."
    assert scrub(text)[0] == text


def test_email_and_ip():
    text = (
        "Escriba a contacto@municipalidad.cl o a juan.perez+boletin@correo.uc.cl; "
        "el servidor 192.168.0.1 respondió, pero 256.1.1.1 y la versión 1.2.3.4.5 no son IP."
    )
    out, counts = scrub(text)
    assert out == (
        "Escriba a <EMAIL> o a <EMAIL>; "
        "el servidor <IP> respondió, pero 256.1.1.1 y la versión 1.2.3.4.5 no son IP."
    )
    assert counts == {"EMAIL": 2, "RUT": 0, "PHONE": 0, "IP": 1}


def test_email_is_matched_before_numbers_inside_it():
    out, counts = scrub("Mi correo es rut12345678-5@empresa.cl.")
    assert out == "Mi correo es <EMAIL>."
    assert counts["RUT"] == 0


def test_prices_dates_and_plain_numbers_are_untouched():
    text = "El 18-09-2024 el kilo costaba $9.990 y la población era 19.458.310 personas."
    assert scrub(text)[0] == text


def test_scrubber_in_pipeline_counts_replacements():
    docs = [
        Document("a", "RUT 12.345.678-5 y correo ana@ejemplo.cl"),
        Document("b", "Sin datos personales."),
    ]
    pipe = Pipeline([PIIScrubber()])
    out = list(pipe.run(docs))
    assert out[0].text == "RUT <RUT> y correo <EMAIL>"
    assert out[1] is docs[1]
    stats = pipe.report()["steps"][0]
    assert stats["modified"] == 1
    assert stats["counters"] == {"EMAIL": 1, "RUT": 1}


def test_scrub_is_idempotent():
    text = "Contacto: +56 9 8765 4321, ana@ejemplo.cl, 12.345.678-5, 10.0.0.1"
    once, _ = scrub(text)
    twice, counts = scrub(once)
    assert once == twice
    assert sum(counts.values()) == 0
