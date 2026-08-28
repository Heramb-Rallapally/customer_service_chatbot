"""Ports that let retrieval work with mocks and OCI/Oracle implementations."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from src.models import KnowledgeDocument, RetrievalResult

from .filters import RetrievalFilters


class EmbeddingService(Protocol):
    """Creates dense vectors for text."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Stores document vectors and returns scored nearest neighbours."""

    def upsert(
        self, documents: Sequence[KnowledgeDocument], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    def similarity_search(
        self, query_embedding: Sequence[float], *, k: int, filters: Optional[RetrievalFilters]
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class TextIndexingVectorStore(Protocol):
    """Optional port for stores that embed text during their native insertion API."""

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None: ...
