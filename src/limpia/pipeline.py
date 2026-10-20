"""Composable processing steps with per-step accounting.

A curation run is only auditable if every removed document can be attributed
to a step and a reason. Steps therefore never drop documents themselves; they
return a decision, and the pipeline does the bookkeeping. This keeps the
counts consistent by construction (``kept + removed == seen`` for every step
and ``seen`` of a step equals ``kept`` of the previous one), which the tests
check as an invariant.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from limpia import __version__
from limpia.document import Document


class Filter(ABC):
    """A step that keeps or removes a document without changing it."""

    name: str = "filter"

    @abstractmethod
    def check(self, doc: Document) -> str | None:
        """Return ``None`` to keep the document, or a short removal reason."""


class Transform(ABC):
    """A step that may rewrite a document and reports what it changed."""

    name: str = "transform"

    @abstractmethod
    def apply(self, doc: Document) -> tuple[Document, Mapping[str, int]]:
        """Return the new document and counters (e.g. replacements by type)."""


Step = Filter | Transform


@dataclass
class StepStats:
    name: str
    kind: str
    seen: int = 0
    kept: int = 0
    removed: int = 0
    modified: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    counters: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "seen": self.seen,
            "kept": self.kept,
            "removed": self.removed,
            "modified": self.modified,
            "reasons": dict(sorted(self.reasons.items())),
            "counters": dict(sorted(self.counters.items())),
        }


class Pipeline:
    """Run documents through an ordered sequence of filters and transforms.

    ``run`` is a generator so that arbitrarily large inputs stream with
    constant memory (stateful steps such as exact dedup aside). The report is
    deliberately free of timestamps and timings so that two runs over the same
    input produce byte-identical reports, which makes them diffable in review.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {names}")
        self.steps = list(steps)
        self.stats = [
            StepStats(step.name, "filter" if isinstance(step, Filter) else "transform")
            for step in self.steps
        ]
        self.input_docs = 0
        self.output_docs = 0

    def process(self, doc: Document) -> Document | None:
        """Process one document; return it (possibly rewritten) or ``None``."""
        self.input_docs += 1
        for step, stats in zip(self.steps, self.stats, strict=True):
            stats.seen += 1
            if isinstance(step, Filter):
                reason = step.check(doc)
                if reason is not None:
                    stats.removed += 1
                    stats.reasons[reason] += 1
                    return None
            else:
                new_doc, counts = step.apply(doc)
                stats.counters.update({k: v for k, v in counts.items() if v})
                if new_doc != doc:
                    stats.modified += 1
                doc = new_doc
            stats.kept += 1
        self.output_docs += 1
        return doc

    def run(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            out = self.process(doc)
            if out is not None:
                yield out

    def report(self) -> dict[str, Any]:
        return {
            "limpia_version": __version__,
            "input_docs": self.input_docs,
            "output_docs": self.output_docs,
            "steps": [s.to_dict() for s in self.stats],
        }

    def write_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
