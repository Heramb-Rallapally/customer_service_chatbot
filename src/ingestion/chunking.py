"""Deterministic, overlap-aware text chunking."""

from __future__ import annotations

from .models import ChunkingConfig


def chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    """Split normalized text into stable chunks, preferring nearby whitespace."""

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        limit = min(start + config.chunk_size, length)
        end = limit
        if limit < length:
            boundary = text.rfind(" ", start + 1, limit + 1)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        # A word boundary can create a very short first segment; always move
        # forward even when its length is less than the requested overlap.
        start = max(end - config.chunk_overlap, start + 1)
        while start < length and text[start].isspace():
            start += 1
    return chunks
