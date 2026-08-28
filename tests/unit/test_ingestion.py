"""Unit tests for the Member 1 knowledge ingestion pipeline."""

import json

import pytest

from src.ingestion import (
    ChunkingConfig,
    IngestionError,
    IngestionRecord,
    KnowledgeIngestionPipeline,
    MetadataExtractor,
    SourceType,
    UnsupportedSourceError,
    load_file,
)
from src.ingestion.chunking import chunk_text
from src.ingestion.cleaning import normalize_text


def test_normalize_text_removes_excess_whitespace_without_losing_paragraphs() -> None:
    assert normalize_text("  VPN\tconnection  \r\n\r\n\r\n Retry it.  ") == "VPN connection\n\nRetry it."


def test_chunking_is_deterministic_and_preserves_requested_overlap() -> None:
    config = ChunkingConfig(chunk_size=20, chunk_overlap=5)
    text = "alpha bravo charlie delta echo foxtrot golf hotel"

    first = chunk_text(text, config)

    assert first == chunk_text(text, config)
    assert len(first) > 1
    assert all(chunk for chunk in first)
    assert "delta" in first[0] or "delta" in first[1]


def test_chunking_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_ingestion_preserves_metadata_and_extracts_shared_fields() -> None:
    pipeline = KnowledgeIngestionPipeline(metadata_extractor=MetadataExtractor(("Oracle VPN",)))

    documents = pipeline.ingest_text(
        source="vpn-guide.md",
        content="Oracle VPN version 5.2 has an authentication timeout. Restart the client as a workaround.",
        source_type=SourceType.TROUBLESHOOTING_GUIDE,
        metadata={"authoritative": True, "ticket_status": "resolved"},
    )

    document = documents[0]
    assert document.product == "Oracle VPN"
    assert document.version == "5.2"
    assert document.issue_type == "connectivity"
    assert document.resolution_category == "workaround"
    assert document.metadata == {
        "authoritative": True,
        "ticket_status": "resolved",
        "source_type": "troubleshooting_guide",
        "chunk_index": 0,
        "chunk_count": 1,
    }


def test_supplied_metadata_takes_precedence_over_heuristics() -> None:
    document = KnowledgeIngestionPipeline().ingest_text(
        source="ticket-1",
        content="The service is unavailable; restart it.",
        source_type=SourceType.HISTORICAL_TICKET,
        metadata={"product": "Payments", "severity": "low", "version": "9.1"},
    )[0]

    assert document.product == "Payments"
    assert document.severity.value == "LOW"
    assert document.version == "9.1"


def test_empty_or_invalid_records_are_rejected() -> None:
    pipeline = KnowledgeIngestionPipeline()
    with pytest.raises(IngestionError, match="no usable content"):
        pipeline.ingest_text(source="empty", content=" \n\t ")
    with pytest.raises(IngestionError, match="IngestionRecord"):
        pipeline.ingest_records(["not-a-record"])  # type: ignore[list-item]


def test_pipeline_output_is_deterministic() -> None:
    pipeline = KnowledgeIngestionPipeline(ChunkingConfig(chunk_size=24, chunk_overlap=4))
    record = IngestionRecord(source="faq.txt", content="One two three four five six seven eight nine ten.", source_type=SourceType.FAQ)

    first = pipeline.ingest_records([record])
    second = pipeline.ingest_records([record])

    assert first == second
    assert [document.id for document in first] == [document.id for document in second]


def test_load_text_json_and_csv_sources(tmp_path) -> None:
    text_path = tmp_path / "video-transcript.txt"
    text_path.write_text("Speaker: restart the client", encoding="utf-8")
    json_path = tmp_path / "tickets.json"
    json_path.write_text(json.dumps([{"content": "Login fails", "product": "Portal", "severity": "HIGH"}]), encoding="utf-8")
    csv_path = tmp_path / "faq.csv"
    csv_path.write_text("content,product\nHow do I reset?,Portal\n", encoding="utf-8")

    text_record = load_file(text_path)[0]
    json_record = load_file(json_path)[0]
    csv_record = load_file(csv_path)[0]

    assert text_record.source_type is SourceType.VIDEO_TRANSCRIPT
    assert json_record.metadata["product"] == "Portal"
    assert json_record.metadata["severity"] == "HIGH"
    assert csv_record.source_type is SourceType.FAQ
    assert csv_record.metadata["product"] == "Portal"


def test_load_file_reports_invalid_and_unsupported_sources(tmp_path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid", encoding="utf-8")
    binary = tmp_path / "knowledge.pdf"
    binary.write_bytes(b"not parsed")

    with pytest.raises(IngestionError, match="invalid JSON"):
        load_file(bad_json)
    with pytest.raises(UnsupportedSourceError, match="unsupported"):
        load_file(binary)
