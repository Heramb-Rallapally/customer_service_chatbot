"""Credential-free integration coverage for ingestion output sent to indexing."""

from __future__ import annotations

from src.ingestion import (
    IngestionRecord,
    KnowledgeIndexer,
    KnowledgeIngestionPipeline,
    MetadataExtractor,
)


class FakeRetrievalService:
    def __init__(self) -> None:
        self.documents = []

    def index_documents(self, documents: object) -> None:
        self.documents.extend(documents)  # type: ignore[arg-type]


def test_ingestion_output_flows_to_retrieval_indexing_without_conversion() -> None:
    retrieval = FakeRetrievalService()
    indexer = KnowledgeIndexer(
        retrieval,
        ingestion_pipeline=KnowledgeIngestionPipeline(
            metadata_extractor=MetadataExtractor(("Oracle VPN",))
        ),
    )

    documents = indexer.ingest_records_and_index(
        [
            IngestionRecord(
                source="vpn-guide.txt",
                content="Oracle VPN version 5.2 has an authentication timeout.",
            )
        ]
    )

    assert retrieval.documents == documents
    assert retrieval.documents[0] is documents[0]
    assert retrieval.documents[0].id.startswith("knowledge-")
    assert retrieval.documents[0].content == "Oracle VPN version 5.2 has an authentication timeout."
    assert retrieval.documents[0].product == "Oracle VPN"
    assert retrieval.documents[0].version == "5.2"
    assert retrieval.documents[0].issue_type == "connectivity"
