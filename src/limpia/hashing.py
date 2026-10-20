"""Stable 64-bit string hashing.

Python's built-in ``hash`` is salted per process, so it cannot be used for
anything that must be reproducible across runs or machines. BLAKE2b is in the
standard library and is the default; xxHash is faster but optional, and the two
produce different values, so the hasher is an explicit parameter rather than an
implicit "use xxhash if installed" switch that would make outputs depend on the
environment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Literal

Hasher = Literal["blake2b", "xxh64"]


def _blake2b64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def get_hash_fn(hasher: Hasher = "blake2b") -> Callable[[bytes], int]:
    """Return a function mapping bytes to an unsigned 64-bit integer."""
    if hasher == "blake2b":
        return _blake2b64
    if hasher == "xxh64":
        try:
            import xxhash
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("hasher='xxh64' requires `pip install limpia[fast]`") from exc
        return xxhash.xxh64_intdigest
    raise ValueError(f"unknown hasher {hasher!r}; expected 'blake2b' or 'xxh64'")


def hash64(text: str, hasher: Hasher = "blake2b") -> int:
    """Hash a string (UTF-8) to an unsigned 64-bit integer."""
    return get_hash_fn(hasher)(text.encode("utf-8"))
