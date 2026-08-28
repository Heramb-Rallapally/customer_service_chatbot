"""Live OracleVS integration for the exact pinned LangChain Community release.

This test is intentionally opt-in. It performs real inserts/searches against
the caller-provided isolated table and is not a mock substitute.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest


_REQUIRED_ENVIRONMENT = (
    "RUN_ORACLEVS_INTEGRATION",
    "ORACLE_DB_USER",
    "ORACLE_DB_PASSWORD",
    "ORACLE_DB_DSN",
    "ORACLEVS_INTEGRATION_TABLE",
)
_MISSING = [name for name in _REQUIRED_ENVIRONMENT if not os.environ.get(name)]
pytestmark = pytest.mark.skipif(
    bool(_MISSING) or os.environ.get("RUN_ORACLEVS_INTEGRATION") != "1",
    reason="requires RUN_ORACLEVS_INTEGRATION=1 and live Oracle AI Database credentials/table",
)


def test_pinned_oraclevs_add_and_search_round_trip() -> None:
    from langchain_community.vectorstores.oraclevs import OracleVS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.embeddings import Embeddings
    import oracledb

    from src.models import KnowledgeDocument
    from src.retrieval import OracleVSVectorStore, RetrievalFilters, SimilarityMetric

    table_name = os.environ["ORACLEVS_INTEGRATION_TABLE"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", table_name):
        pytest.fail("ORACLEVS_INTEGRATION_TABLE must be a single valid Oracle identifier")

    class TestEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0] if "retrieval probe" in text else [0.0, 1.0, 0.0]

    connection = oracledb.connect(
        user=os.environ["ORACLE_DB_USER"],
        password=os.environ["ORACLE_DB_PASSWORD"],
        dsn=os.environ["ORACLE_DB_DSN"],
    )
    document_id = f"live-{uuid.uuid4()}"
    backend = OracleVS(
        connection,
        TestEmbeddings(),
        table_name,
        distance_strategy=DistanceStrategy.COSINE,
        query="retrieval probe",
    )
    store = OracleVSVectorStore(
        backend,
        metric=SimilarityMetric.COSINE,
        embedding_dimension=3,
    )
    document = KnowledgeDocument(
        id=document_id,
        content="OracleVS retrieval probe",
        source="live-integration",
        product="Oracle VPN",
    )
    try:
        store.upsert([document], [[1.0, 0.0, 0.0]])
        results = store.similarity_search(
            [1.0, 0.0, 0.0], k=1, filters=RetrievalFilters(product="oracle vpn")
        )
        assert results[0].document_id == document_id
        assert results[0].metadata["source"] == "live-integration"
        assert 0.0 <= results[0].score <= 1.0
    finally:
        backend.delete(ids=[document_id])
        connection.close()
