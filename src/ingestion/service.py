"""Orchestration for transforming raw knowledge into shared documents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from src.models import KnowledgeDocument

from .chunking import chunk_text
from .cleaning import normalize_text
from .exceptions import IngestionError
from .loaders import load_file
from .metadata import MetadataExtractor
from .models import ChunkingConfig, IngestionRecord, SourceType


class KnowledgeIngestionPipeline:
    """A deterministic, local pipeline for Member 1's knowledge sources."""

    def __init__(
        self,
        chunking: ChunkingConfig | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ) -> None:
        self._chunking = chunking or ChunkingConfig()
        self._metadata_extractor = metadata_extractor or MetadataExtractor()

    def ingest_records(self, records: Iterable[IngestionRecord]) -> list[KnowledgeDocument]:
        """Clean, enrich, chunk, and validate raw records as shared documents."""

        documents: list[KnowledgeDocument] = []
        for record in records:
            if not isinstance(record, IngestionRecord):
                raise IngestionError("records must contain IngestionRecord instances")
            normalized = normalize_text(record.content)
            if not normalized:
                raise IngestionError(f"source contains no usable content: {record.source}")
            chunks = chunk_text(normalized, self._chunking)
            extracted = self._metadata_extractor.extract(record, normalized)
            for index, chunk in enumerate(chunks):
                metadata = {
                    **record.metadata,
                    "source_type": record.source_type.value,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                }
                documents.append(
                    KnowledgeDocument(
                        id=self._document_id(record.source, index, chunk),
                        content=chunk,
                        source=record.source,
                        product=extracted["product"],
                        issue_type=extracted["issue_type"],
                        severity=extracted["severity"],
                        resolution_category=extracted["resolution_category"],
                        version=extracted["version"],
                        metadata=metadata,
                    )
                )
        return documents

    def ingest_text(
        self,
        *,
        source: str,
        content: str,
        source_type: SourceType = SourceType.PRODUCT_DOCUMENTATION,
        metadata: dict[str, object] | None = None,
    ) -> list[KnowledgeDocument]:
        """Ingest one text source through the same validation path as file inputs."""

        return self.ingest_records(
            [IngestionRecord(source=source, content=content, source_type=source_type, metadata=metadata or {})]
        )

    def ingest_file(self, path: str | Path, source_type: SourceType | None = None) -> list[KnowledgeDocument]:
        """Load and ingest a local supported source file."""

        return self.ingest_records(load_file(path, source_type))

    @staticmethod
    def _document_id(source: str, chunk_index: int, content: str) -> str:
        material = f"{source}\x1f{chunk_index}\x1f{content}".encode("utf-8")
        return f"knowledge-{hashlib.sha256(material).hexdigest()[:20]}"
