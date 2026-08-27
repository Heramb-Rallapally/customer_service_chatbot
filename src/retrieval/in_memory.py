"""Deterministic retrieval implementations for local development and tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Optional

from src.models import KnowledgeDocument, RetrievalResult

from .exceptions import VectorStoreError
from .filters import RetrievalFilters
from .metrics import SimilarityMetric, normalized_similarity


class HashEmbeddingService:
    """Stable, dependency-free embeddings for tests; not for production retrieval."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class InMemoryVectorStore:
    """Deterministic vector store implementing the production store port.

    It is intended for tests and local development. Scores are normalized to
    `[0, 1]`, where higher scores are always better.
    """

    def __init__(self, metric: SimilarityMetric = SimilarityMetric.COSINE) -> None:
        self.metric = SimilarityMetric(metric)
        self._entries: dict[str, tuple[KnowledgeDocument, list[float]]] = {}

    def upsert(
        self, documents: Sequence[KnowledgeDocument], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise VectorStoreError("documents and embeddings must have the same length")
        for document, embedding in zip(documents, embeddings):
            if not embedding:
                raise VectorStoreError("document embeddings must not be empty")
            self._entries[document.id] = (document, list(embedding))

    def similarity_search(
        self, query_embedding: Sequence[float], *, k: int, filters: Optional[RetrievalFilters]
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not query_embedding:
            raise VectorStoreError("query embedding must not be empty")
        results: list[RetrievalResult] = []
        for document, embedding in self._entries.values():
            if len(embedding) != len(query_embedding):
                raise VectorStoreError("query and stored embedding dimensions do not match")
            metadata = _document_metadata(document)
            if filters and not _matches(metadata, filters):
                continue
            score = _similarity(query_embedding, embedding, self.metric)
            results.append(
                RetrievalResult(document_id=document.id, content=document.content, score=score, metadata=metadata)
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:k]


def _document_metadata(document: KnowledgeDocument) -> dict[str, object]:
    metadata: dict[str, object] = dict(document.metadata)
    for field in ("source", "product", "issue_type", "severity", "resolution_category", "version"):
        value = getattr(document, field)
        if value is not None:
            metadata[field] = value.value if hasattr(value, "value") else value
    return metadata


def _matches(metadata: dict[str, object], filters: RetrievalFilters) -> bool:
    return all(metadata.get(name) == value for name, value in filters.as_metadata().items())


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def _similarity(
    left: Sequence[float], right: Sequence[float], metric: SimilarityMetric
) -> float:
    if metric is SimilarityMetric.COSINE:
        return normalized_similarity(_cosine_similarity(left, right), metric)
    if metric is SimilarityMetric.DOT:
        return normalized_similarity(sum(x * y for x, y in zip(left, right)), metric)
    # Convert distance to a bounded similarity so callers can consistently
    # interpret greater scores as better results.
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right)))
    return 1.0 / (1.0 + distance)
