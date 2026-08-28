"""Application service that coordinates embedding and vector search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from src.models import KnowledgeDocument, RetrievalResult

from .filters import RetrievalFilters
from .interfaces import EmbeddingService, VectorStore


class RetrievalService:
    """Indexes knowledge and retrieves grounded context through defined ports."""

    def __init__(self, embeddings: EmbeddingService, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        if not documents:
            return
        embeddings = self._embeddings.embed_documents([document.content for document in documents])
        if len(embeddings) != len(documents):
            raise ValueError("embedding service returned an unexpected number of vectors")
        self._vector_store.upsert(documents, embeddings)

    def retrieve(
        self, query: str, *, k: int = 5, filters: Optional[RetrievalFilters] = None
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self._vector_store.similarity_search(
            self._embeddings.embed_query(query), k=k, filters=filters
        )

    def search(
        self,
        *,
        query: str,
        filters: Mapping[str, str],
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        """Retrieve results through the Member 3 conversation-facing port.

        Scores are normalized to `[0, 1]`, with higher scores indicating more
        relevant results, irrespective of the configured vector metric.
        """

        return self.retrieve(
            query,
            k=top_k,
            filters=RetrievalFilters.from_conversation_mapping(filters),
        )
