"""Opt-in real Ollama + Oracle 23ai RAG integration test."""

from __future__ import annotations

import hashlib
import os
import re
import uuid

import pytest


_REQUIRED_ENVIRONMENT = (
    "RUN_OLLAMA_ORACLE_INTEGRATION",
    "ORACLE_DB_USER",
    "ORACLE_DB_PASSWORD",
    "ORACLE_DB_DSN",
    "ORACLEVS_INTEGRATION_TABLE",
)
_MISSING = [name for name in _REQUIRED_ENVIRONMENT if not os.environ.get(name)]
pytestmark = pytest.mark.skipif(
    bool(_MISSING) or os.environ.get("RUN_OLLAMA_ORACLE_INTEGRATION") != "1",
    reason="requires explicit live Ollama and Oracle 23ai integration configuration",
)


def test_ollama_embeddings_oraclevs_retrieval_and_generation() -> None:
    """Exercise real indexing, retrieval, and grounded local generation."""

    from src.app import create_application
    from src.config import Settings
    from src.models import KnowledgeDocument

    table_name = os.environ["ORACLEVS_INTEGRATION_TABLE"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", table_name):
        pytest.fail("ORACLEVS_INTEGRATION_TABLE must be a single valid Oracle identifier")

    document_id = f"ollama-live-{uuid.uuid4()}"
    settings = Settings(
        llm_provider="ollama",
        embedding_provider="ollama",
        llm_model="llama3.2:3b",
        embedding_model="nomic-embed-text",
        embedding_dimension=768,
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        oracle_db_user=os.environ["ORACLE_DB_USER"],
        oracle_db_password=os.environ["ORACLE_DB_PASSWORD"],
        oracle_db_dsn=os.environ["ORACLE_DB_DSN"],
        oracle_vs_table=table_name,
    )
    services = create_application(settings=settings)
    document = KnowledgeDocument(
        id=document_id,
        content=(
            "For the live integration probe, reset the blue support token in "
            "Oracle VPN settings, then reconnect."
        ),
        source="ollama-oracle-live-probe",
        product="Oracle VPN",
        issue_type="authentication",
        version="5.2",
    )
    processed_id = hashlib.sha256(document_id.encode()).hexdigest()[:16].upper()
    try:
        services.knowledge_indexer.index_documents([document])
        results = services.retrieval_service.search(
            query="Oracle VPN 5.2 blue support token authentication",
            filters={"product": "Oracle VPN", "version": "5.2"},
            top_k=3,
        )
        assert any(result.document_id == document_id for result in results)

        conversation_id = f"ollama-e2e-{uuid.uuid4()}"
        clarification = services.conversation_engine.handle_message(
            conversation_id=conversation_id,
            user_id="live-test-user",
            user_message="Oracle VPN version 5.2 authentication fails with the blue support token.",
        )
        assert clarification.message.strip()
        response = services.conversation_engine.handle_message(
            conversation_id=conversation_id,
            user_id="live-test-user",
            user_message="Authentication failed when I used the blue support token.",
        )
        assert response.message.strip()
        assert response.citations
        assert any(citation.document_id == document_id for citation in response.citations)
    finally:
        with services.oracle_connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {table_name} WHERE id = HEXTORAW(:document_id)",
                document_id=processed_id,
            )
        services.oracle_connection.commit()
        services.close()
