"""The unit of data that flows through a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A text with a stable identifier and free-form metadata.

    Documents are immutable: a transform returns a new instance instead of
    mutating its input. This makes it safe to keep references to the original
    (for example, to diff before and after PII scrubbing) and rules out a
    class of bugs where one step silently changes what an earlier step saw.

    ``metadata`` holds every input field other than the id and text, so a
    read-process-write cycle round-trips fields that limpia does not know about
    (URLs, crawl dates, language scores).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_text(self, text: str) -> Document:
        return replace(self, text=text)

    def with_metadata(self, **updates: Any) -> Document:
        return replace(self, metadata={**self.metadata, **updates})
