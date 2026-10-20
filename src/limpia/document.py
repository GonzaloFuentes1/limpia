"""The unit of data that flows through a pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A text with a stable identifier and free-form metadata.

    Documents are shallowly immutable: the fields cannot be reassigned, and
    ``metadata`` is a read-only view of a private copy of the mapping passed
    in, so neither the caller's dict nor another document can change it. A
    transform returns a new instance instead of mutating its input, which makes
    it safe to keep references to the original (for example, to diff before
    and after PII scrubbing). Values nested inside ``metadata`` (lists, dicts)
    are not copied and remain mutable; steps must not modify them in place.

    ``metadata`` holds every input field other than the id and text, so a
    read-process-write cycle round-trips fields that limpia does not know about
    (URLs, crawl dates, language scores).
    """

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __hash__(self) -> int:
        # Equal documents have equal id and text, so this is consistent with
        # the generated __eq__; metadata (a mapping) is not hashable.
        return hash((self.id, self.text))

    def __reduce__(self) -> tuple[Any, ...]:
        # MappingProxyType cannot be pickled; rebuild from a plain dict.
        return (Document, (self.id, self.text, dict(self.metadata)))

    def with_text(self, text: str) -> Document:
        return replace(self, text=text)

    def with_metadata(self, **updates: Any) -> Document:
        return replace(self, metadata={**self.metadata, **updates})
