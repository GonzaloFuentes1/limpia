"""Per-language threshold configuration.

Thresholds live in YAML rather than in code so that a reviewer can audit, for
each number, where it came from and why it applies to the language at hand.
The loader enforces that contract: every entry must declare ``value``,
``source`` (a key of the file's ``sources`` table) and ``rationale``, and
unknown or missing keys are errors rather than silently ignored, because a
typo in a threshold name would otherwise fall back to a default unnoticed.

Thresholds typed ``float | None`` may be ``null`` in the YAML, which disables
that rule; the entry still needs a source and a rationale saying why.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GopherQualityConfig:
    min_words: int
    max_words: int
    min_mean_word_length: float
    max_mean_word_length: float
    max_symbol_word_ratio: float
    max_bullet_lines_ratio: float
    max_ellipsis_lines_ratio: float
    min_alpha_words_ratio: float
    min_stopwords: int
    bullet_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class GopherRepetitionConfig:
    max_dup_line_frac: float | None
    max_dup_para_frac: float | None
    max_dup_line_char_frac: float | None
    max_dup_para_char_frac: float | None
    max_top_ngram_char_frac: dict[int, float]
    max_dup_ngram_char_frac: dict[int, float]


@dataclass(frozen=True)
class FineWebConfig:
    min_terminal_punct_ratio: float | None
    max_short_line_ratio: float | None
    short_line_chars: int
    max_dup_line_char_frac: float | None
    max_newline_word_ratio: float | None
    terminal_punctuation: tuple[str, ...]


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    stopwords: frozenset[str]
    gopher_quality: GopherQualityConfig
    gopher_repetition: GopherRepetitionConfig
    fineweb: FineWebConfig
    sources: dict[str, dict[str, str]]
    provenance: dict[str, dict[str, dict[str, Any]]]


_SECTIONS: dict[str, type] = {
    "gopher_quality": GopherQualityConfig,
    "gopher_repetition": GopherRepetitionConfig,
    "fineweb": FineWebConfig,
}


def available_languages() -> list[str]:
    root = resources.files("limpia") / "resources" / "configs"
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def load_stopwords(name: str) -> frozenset[str]:
    """Load a bundled stopword list (``#`` comments and blank lines ignored)."""
    path = resources.files("limpia") / "resources" / "stopwords" / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(w.strip().lower() for w in lines if w.strip() and not w.startswith("#"))


def _coerce(value: Any, annotation: str) -> Any:
    if annotation.endswith(" | None"):
        if value is None:
            return None
        annotation = annotation.removesuffix(" | None")
    elif value is None:
        raise ValueError("null is not allowed here")
    if annotation.startswith("tuple"):
        return tuple(value)
    if annotation.startswith("dict[int"):
        return {int(k): float(v) for k, v in value.items()}
    if annotation == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"expected an integer, got {value!r}")
        return value
    if annotation == "float":
        return float(value)
    return value


def _build_section(name: str, raw: dict[str, Any], sources: dict[str, Any]) -> Any:
    cls = _SECTIONS[name]
    expected = {f.name: str(f.type) for f in fields(cls)}
    missing = expected.keys() - raw.keys()
    unknown = raw.keys() - expected.keys()
    if missing or unknown:
        raise ValueError(f"[{name}] missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or {"value", "source", "rationale"} - entry.keys():
            raise ValueError(f"[{name}.{key}] needs 'value', 'source' and 'rationale'")
        if entry["source"] not in sources:
            raise ValueError(f"[{name}.{key}] unknown source {entry['source']!r}")
        if not str(entry["rationale"]).strip():
            raise ValueError(f"[{name}.{key}] empty rationale")
        try:
            kwargs[key] = _coerce(entry["value"], expected[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[{name}.{key}] {exc}") from exc
    return cls(**kwargs)


def parse_language_config(raw: dict[str, Any]) -> LanguageConfig:
    sources = raw.get("sources") or {}
    for key, src in sources.items():
        if not {"citation", "url"} <= set(src):
            raise ValueError(f"source {key!r} needs 'citation' and 'url'")
    sections = {name: _build_section(name, raw.get(name) or {}, sources) for name in _SECTIONS}
    provenance = {
        name: {
            key: {"value": e["value"], "source": e["source"], "rationale": e["rationale"]}
            for key, e in (raw.get(name) or {}).items()
        }
        for name in _SECTIONS
    }
    return LanguageConfig(
        language=raw["language"],
        stopwords=load_stopwords(raw["stopwords"]),
        sources=sources,
        provenance=provenance,
        **sections,
    )


def load_language_config(lang_or_path: str | Path) -> LanguageConfig:
    """Load a bundled config by language code (``es``, ``pt``) or a YAML path."""
    path = Path(lang_or_path)
    if path.suffix in {".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
    else:
        if str(lang_or_path) not in available_languages():
            raise ValueError(
                f"no bundled config for {lang_or_path!r}; available: {available_languages()}"
            )
        res = resources.files("limpia") / "resources" / "configs" / f"{lang_or_path}.yaml"
        text = res.read_text(encoding="utf-8")
    return parse_language_config(yaml.safe_load(text))
