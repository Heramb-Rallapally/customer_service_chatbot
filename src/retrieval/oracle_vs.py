"""Adapter for the pinned ``langchain-community`` OracleVS API.

This module is deliberately implemented against ``langchain-community==0.3.31``.
That release provides ``add_texts`` and
``similarity_search_by_vector_with_relevance_scores``.  Despite the latter
method's name, its OracleVS implementation returns ``vector_distance`` values
ordered from lowest to highest.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from src.models import KnowledgeDocument, RetrievalResult

from .exceptions import VectorStoreError
from .filters import RetrievalFilters
from .in_memory import _document_metadata, _matches
from .metrics import OracleScoreSemantics, SimilarityMetric, normalized_oracle_score


class OracleVSVectorStore:
    """Adapt a real LangChain ``OracleVS`` instance to the retrieval port.

    Production uses COSINE distance.  The matching Oracle vector index must
    use the same distance strategy.  ``embedding_dimension`` is optional for
    backward-compatible construction, but production callers should set it so
    query and indexing vectors fail early on a dimension mismatch.

    LangChain Community 0.3.31's metadata filter applies ``value in filter``
    to scalar strings, which is not exact-match filtering.  Consequently this
    adapter intentionally retrieves unfiltered candidates and filters them
    locally with normalized exact comparisons.
    """

    def __init__(
        self,
        backend: Any,
        *,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        score_semantics: OracleScoreSemantics = OracleScoreSemantics.DISTANCE,
        embedding_dimension: Optional[int] = None,
        max_candidate_fetch: Optional[int] = None,
    ) -> None:
        self._backend = backend
        self.metric = SimilarityMetric(metric)
        self.score_semantics = OracleScoreSemantics(score_semantics)
        if self.score_semantics is not OracleScoreSemantics.DISTANCE:
            raise ValueError(
                "langchain-community 0.3.31 OracleVS returns vector distances, not similarities"
            )
        if embedding_dimension is not None and embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if max_candidate_fetch is not None and max_candidate_fetch < 1:
            raise ValueError("max_candidate_fetch must be positive")
        self._embedding_dimension = embedding_dimension
        self._max_candidate_fetch = max_candidate_fetch
        self._validate_backend_metric()

    def upsert(
        self, documents: Sequence[KnowledgeDocument], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Insert documents using OracleVS 0.3.31's supported ``add_texts`` API.

        OracleVS does not offer an ``add_embeddings`` API in the pinned release:
        it embeds ``texts`` using the embedding function supplied at OracleVS
        construction.  The vectors received through the shared port are still
        validated here, ensuring the query-side embedder and document-side
        embedder are configured for the same dimension before insertion.
        """

        if len(documents) != len(embeddings):
            raise VectorStoreError("documents and embeddings must have the same length")
        if not documents:
            return
        self._validate_document_ids(documents)
        for embedding in embeddings:
            self._validate_embedding(embedding, label="document")
        self._add_texts(documents)

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        """Index through OracleVS's native text API with one embedding pass.

        ``langchain-community==0.3.31`` has no API for supplied document
        vectors. Its supported ``add_texts`` method embeds with the configured
        ``OCIEmbeddingService`` and stores those exact vectors. The OCI adapter
        continues to validate configured dimensions and finite values.
        """

        if not documents:
            return
        self._validate_document_ids(documents)
        self._add_texts(documents)

    def _add_texts(self, documents: Sequence[KnowledgeDocument]) -> None:
        method = getattr(self._backend, "add_texts", None)
        if method is None:
            raise VectorStoreError(
                "OracleVS backend must support add_texts (langchain-community 0.3.31)"
            )
        metadatas = self._metadata_for_documents(documents)
        try:
            method(
                texts=[document.content for document in documents],
                metadatas=metadatas,
                ids=[document.id for document in documents],
            )
        except Exception as exc:
            raise VectorStoreError("OracleVS document insertion failed") from exc

    @staticmethod
    def _validate_document_ids(documents: Sequence[KnowledgeDocument]) -> None:
        duplicate_ids = _duplicates(document.id for document in documents)
        if duplicate_ids:
            raise VectorStoreError("duplicate document IDs in batch: " + ", ".join(duplicate_ids))

    @staticmethod
    def _metadata_for_documents(documents: Sequence[KnowledgeDocument]) -> list[dict[str, object]]:
        metadata = [
            {**_document_metadata(document), "document_id": document.id}
            for document in documents
        ]
        try:
            for value in metadata:
                json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise VectorStoreError(
                "OracleVS document metadata must be JSON serializable"
            ) from exc
        return metadata

    def similarity_search(
        self, query_embedding: Sequence[float], *, k: int, filters: Optional[RetrievalFilters]
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        self._validate_embedding(query_embedding, label="query")
        method = getattr(
            self._backend, "similarity_search_by_vector_with_relevance_scores", None
        )
        if method is None:
            raise VectorStoreError(
                "OracleVS backend must support similarity_search_by_vector_with_relevance_scores "
                "(langchain-community 0.3.31)"
            )

        # With a filter, double candidate count until there are enough matches
        # or OracleVS has returned every available candidate.  There is no
        # arbitrary small candidate window that could silently hide a match.
        candidate_k = k
        while True:
            if self._max_candidate_fetch is not None:
                candidate_k = min(candidate_k, self._max_candidate_fetch)
            try:
                pairs = method(embedding=list(query_embedding), k=candidate_k)
            except Exception as exc:
                raise VectorStoreError("OracleVS similarity search failed") from exc
            results = self._convert_pairs(pairs, filters)
            if not filters or len(results) >= k or len(pairs) < candidate_k:
                return sorted(results, key=lambda result: result.score, reverse=True)[:k]
            if self._max_candidate_fetch is not None and candidate_k >= self._max_candidate_fetch:
                raise VectorStoreError(
                    "OracleVS local filtering reached max_candidate_fetch before finding enough matches"
                )
            candidate_k *= 2

    def _convert_pairs(
        self, pairs: Sequence[tuple[Any, Any]], filters: Optional[RetrievalFilters]
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for document, raw_distance in pairs:
            metadata = dict(getattr(document, "metadata", {}) or {})
            if filters and not _matches(metadata, filters):
                continue
            document_id = str(metadata.get("document_id") or metadata.get("id") or "")
            if not document_id:
                raise VectorStoreError("OracleVS result is missing document_id metadata")
            content = getattr(document, "page_content", None)
            if not isinstance(content, str) or not content:
                raise VectorStoreError("OracleVS result is missing document content")
            try:
                raw_distance = float(raw_distance)
                if not math.isfinite(raw_distance):
                    raise ValueError("raw vector distance must be finite")
                score = normalized_oracle_score(raw_distance, self.metric, self.score_semantics)
            except (TypeError, ValueError) as exc:
                raise VectorStoreError("OracleVS returned an invalid vector distance") from exc
            results.append(
                RetrievalResult(
                    document_id=document_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def _validate_backend_metric(self) -> None:
        """Reject an injected OracleVS configured for a different metric."""

        backend_metric = getattr(self._backend, "distance_strategy", None)
        if backend_metric is None:
            return  # Allows mock ports; real OracleVS exposes this attribute.
        backend_name = getattr(backend_metric, "name", str(backend_metric)).upper()
        expected = {
            SimilarityMetric.COSINE: "COSINE",
            SimilarityMetric.EUCLIDEAN: "EUCLIDEAN_DISTANCE",
            SimilarityMetric.DOT: "DOT_PRODUCT",
        }[self.metric]
        if backend_name != expected:
            raise ValueError(
                f"OracleVS distance strategy {backend_name!r} does not match configured metric {self.metric.value!r}"
            )

    def _validate_embedding(self, embedding: Sequence[float], *, label: str) -> None:
        if not embedding:
            raise VectorStoreError(f"{label} embedding must not be empty")
        try:
            values_are_finite = all(math.isfinite(float(value)) for value in embedding)
        except (TypeError, ValueError) as exc:
            raise VectorStoreError(f"{label} embedding must contain numeric values") from exc
        if not values_are_finite:
            raise VectorStoreError(f"{label} embedding must contain only finite values")
        dimension = len(embedding)
        if self._embedding_dimension is None:
            self._embedding_dimension = dimension
        elif dimension != self._embedding_dimension:
            raise VectorStoreError(
                f"{label} embedding dimension {dimension} does not match expected "
                f"dimension {self._embedding_dimension}"
            )


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
