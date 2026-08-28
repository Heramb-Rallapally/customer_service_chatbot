"""Connect shared ingestion output to the existing retrieval indexing service."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from src.models import KnowledgeDocument

from .exceptions import IndexingError
from .models import IngestionRecord, SourceType
from .service import KnowledgeIngestionPipeline


@runtime_checkable
class DocumentIndexer(Protocol):
    """Narrow indexing boundary implemented by ``RetrievalService``."""

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        """Index one batch of shared knowledge documents."""


class KnowledgeIndexer:
    """Index ingestion output through ``RetrievalService`` without owning embeddings.

    The same ``KnowledgeDocument`` instances are passed to retrieval.  This
    preserves their deterministic IDs, content, structured fields, and extra
    metadata for the retrieval adapter to store.  Retrieval/database failures
    intentionally propagate unchanged so callers can handle the real cause.
    """

    def __init__(
        self,
        retrieval_service: DocumentIndexer,
        *,
        ingestion_pipeline: Optional[KnowledgeIngestionPipeline] = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._ingestion_pipeline = ingestion_pipeline or KnowledgeIngestionPipeline()

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> int:
        """Validate and submit one existing retrieval batch; return its document count."""

        if not documents:
            return 0
        if any(not isinstance(document, KnowledgeDocument) for document in documents):
            raise IndexingError("documents must contain KnowledgeDocument instances")
        self._retrieval_service.index_documents(documents)
        return len(documents)

    def ingest_records_and_index(
        self, records: Iterable[IngestionRecord]
    ) -> list[KnowledgeDocument]:
        """Run the existing record pipeline, then index its shared documents."""

        documents = self._ingestion_pipeline.ingest_records(records)
        self.index_documents(documents)
        return documents

    def ingest_file_and_index(
        self, path: str | Path, source_type: Optional[SourceType] = None
    ) -> list[KnowledgeDocument]:
        """Load, ingest, and index one supported local source file."""

        documents = self._ingestion_pipeline.ingest_file(path, source_type)
        self.index_documents(documents)
        return documents
