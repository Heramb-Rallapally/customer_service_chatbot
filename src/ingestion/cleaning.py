"""Text normalization utilities for ingestion."""

from __future__ import annotations

import re


_WHITESPACE = re.compile(r"[\t\r\f\v ]+")
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize line endings and whitespace without altering document meaning."""

    if not isinstance(text, str):
        raise TypeError("content must be a string")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n"))
    return _EXCESSIVE_NEWLINES.sub("\n\n", normalized).strip()
