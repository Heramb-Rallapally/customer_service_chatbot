"""Normalize provider-backed proactive evidence into shared article references.

No recommendation is invented here: inputs without a usable identifier are discarded.
Source labels retain the evidence category so downstream conversation code can present
historical and knowledge signals with appropriate authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.models.proactive import ArticleReference


def normalize_references(
    references: Iterable[ArticleReference | Mapping[str, Any]], default_source: str
) -> list[ArticleReference]:
    """Convert provider evidence to valid references, preserving an explicit source."""
    normalized: list[ArticleReference] = []
    for reference in references:
        if isinstance(reference, ArticleReference):
            data = reference.model_dump()
        elif isinstance(reference, Mapping):
            data = dict(reference)
        else:
            continue
        article_id = data.get("article_id") or data.get("id") or data.get("document_id")
        if not isinstance(article_id, str) or not article_id.strip():
            continue
        title = data.get("title")
        source = data.get("source") or default_source
        normalized.append(
            ArticleReference(
                article_id=article_id.strip(),
                title=title if isinstance(title, str) else None,
                source=source if isinstance(source, str) else default_source,
            )
        )
    return normalized
