"""Adapter for a configured LangChain OracleVS instance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from src.models import KnowledgeDocument, RetrievalResult

from .exceptions import VectorStoreError
from .filters import RetrievalFilters
from .in_memory import _document_metadata, _matches
from .metrics import SimilarityMetric


class OracleVSVectorStore:
    """Maps Member 2's vector-store port to a LangChain OracleVS backend.

    The caller owns construction of the OracleVS instance and database
    connection. This keeps database lifecycle/schema ownership in `src/db`.
    """

    def __init__(
        self, backend: Any, *, metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> None:
        self._backend = backend
        self.metric = SimilarityMetric(metric)

    def upsert(
        self, documents: Sequence[KnowledgeDocument], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise VectorStoreError("documents and embeddings must have the same length")
        if not hasattr(self._backend, "add_embeddings"):
            raise VectorStoreError("OracleVS backend must support add_embeddings")
        try:
            self._backend.add_embeddings(
                text_embeddings=[(document.content, list(vector)) for document, vector in zip(documents, embeddings)],
                metadatas=[{**_document_metadata(document), "document_id": document.id} for document in documents],
                ids=[document.id for document in documents],
            )
        except Exception as exc:
            raise VectorStoreError("OracleVS document insertion failed") from exc

    def similarity_search(
        self, query_embedding: Sequence[float], *, k: int, filters: Optional[RetrievalFilters]
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        method = getattr(self._backend, "similarity_search_with_score_by_vector", None)
        if method is None:
            raise VectorStoreError("OracleVS backend must support vector similarity search")
        try:
            pairs = method(list(query_embedding), k=k, filter=filters.as_metadata() if filters else None)
        except TypeError:
            # Older LangChain releases lack a metadata-filter keyword.
            try:
                pairs = method(list(query_embedding), k=k)
            except Exception as exc:
                raise VectorStoreError("OracleVS similarity search failed") from exc
        except Exception as exc:
            raise VectorStoreError("OracleVS similarity search failed") from exc
        results = []
        for document, score in pairs:
            metadata = dict(getattr(document, "metadata", {}))
            if filters and not _matches(metadata, filters):
                continue
            document_id = str(metadata.get("document_id") or metadata.get("id") or "")
            if not document_id:
                raise VectorStoreError("OracleVS result is missing document_id metadata")
            results.append(RetrievalResult(document_id=document_id, content=document.page_content, score=float(score), metadata=metadata))
        # OracleVS already returns nearest neighbours in rank order. Preserve that
        # ordering because its score direction is backend/version dependent. The
        # backend must be configured with this adapter's selected metric when it
        # is constructed by the database integration layer.
        return results[:k]
