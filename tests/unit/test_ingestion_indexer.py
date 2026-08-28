"""Credential-free tests for ingestion-to-retrieval indexing orchestration."""

from __future__ import annotations

import importlib

import pytest

from src.ingestion import (
    DocumentIndexer,
    IndexingError,
    IngestionRecord,
    KnowledgeIndexer,
    KnowledgeIngestionPipeline,
    SourceType,
)
from src.models import KnowledgeDocument, Severity


class RecordingRetrievalService:
    def __init__(self) -> None:
        self.batches: list[object] = []

    def index_documents(self, documents: object) -> None:
        self.batches.append(documents)


def documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            id="vpn-5-2-authentication",
            content="Reset the VPN authentication token.",
            source="oracle-vpn-guide",
            product="Oracle VPN",
            version="5.2",
            issue_type="authentication",
            severity=Severity.HIGH,
            resolution_category="configuration_change",
            metadata={"source_type": "troubleshooting_guide", "authoritative": True},
        ),
        KnowledgeDocument(
            id="vpn-5-2-connectivity",
            content="Confirm the network route is available.",
            source="oracle-vpn-guide",
            product="Oracle VPN",
            version="5.2",
            issue_type="connectivity",
            severity=Severity.MEDIUM,
            metadata={"chunk_index": 1, "chunk_count": 2},
        ),
    ]


def test_indexer_delegates_exact_documents_and_preserves_metadata() -> None:
    retrieval = RecordingRetrievalService()
    batch = documents()

    assert isinstance(retrieval, DocumentIndexer)
    indexed = KnowledgeIndexer(retrieval).index_documents(batch)

    assert indexed == 2
    assert retrieval.batches == [batch]
    assert retrieval.batches[0] is batch
    assert [document.id for document in batch] == [
        "vpn-5-2-authentication",
        "vpn-5-2-connectivity",
    ]
    assert batch[0].content == "Reset the VPN authentication token."
    assert batch[0].product == "Oracle VPN"
    assert batch[0].version == "5.2"
    assert batch[0].issue_type == "authentication"
    assert batch[0].severity is Severity.HIGH
    assert batch[0].metadata["authoritative"] is True


def test_empty_input_is_a_noop() -> None:
    retrieval = RecordingRetrievalService()

    assert KnowledgeIndexer(retrieval).index_documents([]) == 0
    assert retrieval.batches == []


def test_non_knowledge_documents_are_rejected() -> None:
    with pytest.raises(IndexingError, match="KnowledgeDocument"):
        KnowledgeIndexer(RecordingRetrievalService()).index_documents(["not-a-document"])  # type: ignore[list-item]


def test_retrieval_failure_propagates_unchanged() -> None:
    failure = RuntimeError("database indexing unavailable")

    class FailingRetrievalService:
        def index_documents(self, _documents: object) -> None:
            raise failure

    with pytest.raises(RuntimeError) as error:
        KnowledgeIndexer(FailingRetrievalService()).index_documents(documents())

    assert error.value is failure


def test_reindexing_delegates_the_same_stable_document_ids_each_time() -> None:
    retrieval = RecordingRetrievalService()
    indexer = KnowledgeIndexer(retrieval)
    batch = documents()

    indexer.index_documents(batch)
    indexer.index_documents(batch)

    assert [[document.id for document in submitted] for submitted in retrieval.batches] == [  # type: ignore[union-attr]
        ["vpn-5-2-authentication", "vpn-5-2-connectivity"],
        ["vpn-5-2-authentication", "vpn-5-2-connectivity"],
    ]


def test_import_is_credential_free() -> None:
    module = importlib.import_module("src.ingestion.indexer")
    importlib.reload(module)

    assert hasattr(module, "KnowledgeIndexer")


def test_ingest_records_and_index_reuses_existing_pipeline() -> None:
    retrieval = RecordingRetrievalService()
    pipeline = KnowledgeIngestionPipeline()
    indexer = KnowledgeIndexer(retrieval, ingestion_pipeline=pipeline)

    output = indexer.ingest_records_and_index(
        [
            IngestionRecord(
                source="vpn-faq.txt",
                content="Oracle VPN version 5.2 has a connection timeout.",
                source_type=SourceType.FAQ,
                metadata={"authoritative": True},
            )
        ]
    )

    assert output == retrieval.batches[0]
    assert output[0].product is None
    assert output[0].version == "5.2"
    assert output[0].issue_type == "connectivity"
    assert output[0].metadata["source_type"] == "faq"


def test_ingest_file_and_index_uses_the_existing_file_pipeline(tmp_path) -> None:
    source = tmp_path / "vpn-guide.txt"
    source.write_text("Oracle VPN version 5.2 needs a connection check.", encoding="utf-8")
    retrieval = RecordingRetrievalService()

    output = KnowledgeIndexer(retrieval).ingest_file_and_index(source)

    assert output == retrieval.batches[0]
    assert output[0].source == str(source)
    assert output[0].content == "Oracle VPN version 5.2 needs a connection check."
