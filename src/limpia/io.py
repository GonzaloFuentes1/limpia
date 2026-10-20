"""Streaming JSONL input and output.

Web-scale corpora do not fit in memory, so readers are generators and writers
consume iterators. Gzip is supported transparently because most public crawl
dumps ship compressed and decompressing them to disk first doubles storage.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from limpia.document import Document


def _open(path: Path, mode: str) -> IO[str]:
    """Open a text stream with ``\\n`` line endings on every platform.

    Gzip output is written with ``mtime=0`` in its header, so the same input
    gives byte-identical ``.jsonl.gz`` files across runs and operating systems.
    """
    if path.suffix == ".gz":
        if mode == "w":
            raw = gzip.GzipFile(path, "wb", mtime=0)
            return io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def read_jsonl(
    path: str | Path, *, text_key: str = "text", id_key: str = "id"
) -> Iterator[Document]:
    """Yield documents from a JSONL file, one JSON object per line.

    Blank lines are skipped. A record without an id gets ``<stem>:<line>``,
    which is stable across runs, so reports and dedup decisions can refer to
    it. Malformed records raise with the line number instead of being skipped:
    silently dropping input would make the kept/removed counts in the report
    disagree with the file.
    """
    path = Path(path)
    stem = path.name.split(".")[0]
    with _open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            text = record.pop(text_key, None)
            if not isinstance(text, str):
                raise ValueError(f"{path}:{lineno}: missing string field {text_key!r}")
            doc_id = record.pop(id_key, None)
            if doc_id is None:
                doc_id = f"{stem}:{lineno}"
            yield Document(id=str(doc_id), text=text, metadata=record)


def document_to_record(
    doc: Document, *, text_key: str = "text", id_key: str = "id"
) -> dict[str, object]:
    return {id_key: doc.id, text_key: doc.text, **doc.metadata}


def write_jsonl(
    docs: Iterable[Document],
    path: str | Path,
    *,
    text_key: str = "text",
    id_key: str = "id",
) -> int:
    """Write documents as JSONL and return how many were written.

    ``ensure_ascii=False`` keeps Spanish text readable and roughly halves the
    size of accented text compared with ``\\u`` escapes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open(path, "w") as fh:
        for doc in docs:
            record = document_to_record(doc, text_key=text_key, id_key=id_key)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
