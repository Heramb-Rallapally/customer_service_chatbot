"""Unit tests for Member 2 retrieval components; no OCI or Oracle required."""

from types import SimpleNamespace

import pytest

from src.config import Settings
from src.models import KnowledgeDocument, Severity
from src.retrieval import (
    HashEmbeddingService,
    InMemoryVectorStore,
    OCIEmbeddingService,
    OracleVSVectorStore,
    RetrievalEvaluationCase,
    RetrievalFilters,
    RetrievalService,
    SimilarityMetric,
    evaluate_retrieval,
)
from src.retrieval.exceptions import EmbeddingError, VectorStoreError


@pytest.fixture
def documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(id="vpn-guide", content="VPN client connection troubleshooting", source="official-guide", product="Oracle VPN", version="5.2", severity=Severity.MEDIUM),
        KnowledgeDocument(id="email-guide", content="Reset your email mailbox password", source="official-guide", product="Oracle Mail", severity=Severity.LOW),
    ]


@pytest.fixture
def service(documents: list[KnowledgeDocument]) -> RetrievalService:
    service = RetrievalService(HashEmbeddingService(), InMemoryVectorStore())
    service.index_documents(documents)
    return service


def test_retrieval_returns_ranked_result(service: RetrievalService) -> None:
    results = service.retrieve("VPN connection", k=1)
    assert [result.document_id for result in results] == ["vpn-guide"]
    assert results[0].metadata["product"] == "Oracle VPN"


def test_retrieval_applies_metadata_filters(service: RetrievalService) -> None:
    results = service.retrieve("connection", filters=RetrievalFilters(product="Oracle Mail"))
    assert [result.document_id for result in results] == ["email-guide"]


@pytest.mark.parametrize("metric", list(SimilarityMetric))
def test_in_memory_store_supports_all_required_similarity_metrics(
    documents: list[KnowledgeDocument], metric: SimilarityMetric
) -> None:
    service = RetrievalService(HashEmbeddingService(), InMemoryVectorStore(metric))
    service.index_documents(documents)
    results = service.search("VPN connection", top_k=1)
    assert [result.document_id for result in results] == ["vpn-guide"]
    assert results[0].score > 0


def test_search_exposes_conversation_facing_signature(service: RetrievalService) -> None:
    assert [result.document_id for result in service.search("VPN", top_k=1)] == ["vpn-guide"]
    with pytest.raises(ValueError, match="k must be at least 1"):
        service.search("VPN", top_k=0)


def test_retrieval_handles_empty_index_and_rejects_empty_query() -> None:
    service = RetrievalService(HashEmbeddingService(), InMemoryVectorStore())
    assert service.retrieve("VPN") == []
    with pytest.raises(ValueError, match="query must not be empty"):
        service.retrieve("   ")


def test_evaluation_calculates_recall_and_mrr(service: RetrievalService) -> None:
    report = evaluate_retrieval(service, [RetrievalEvaluationCase(query="VPN troubleshooting", relevant_document_ids=frozenset({"vpn-guide"}))], k=1)
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_oci_embedding_adapter_wraps_provider_error() -> None:
    class FailingClient:
        def embed_text(self, _request):
            raise RuntimeError("network down")

    adapter = OCIEmbeddingService(FailingClient(), compartment_id="compartment", model_id="model", request_factory=list)
    with pytest.raises(EmbeddingError, match="OCI embedding request failed"):
        adapter.embed_query("vpn")


def test_oci_embedding_adapter_returns_provider_vectors() -> None:
    class Client:
        def embed_text(self, request):
            assert request == ["vpn"]
            return SimpleNamespace(data=SimpleNamespace(embeddings=[[0.1, 0.2]]))

    adapter = OCIEmbeddingService(Client(), compartment_id="compartment", model_id="model", request_factory=list)
    assert adapter.embed_query("vpn") == [0.1, 0.2]


def test_oci_adapter_requires_configuration() -> None:
    with pytest.raises(EmbeddingError, match="OCI_COMPARTMENT_ID"):
        OCIEmbeddingService.from_settings(Settings())


def test_oraclevs_adapter_maps_insert_and_search() -> None:
    class Backend:
        def add_embeddings(self, **kwargs):
            self.inserted = kwargs

        def similarity_search_with_score_by_vector(self, vector, *, k, filter=None):
            assert vector == [0.4, 0.6]
            assert k == 2
            assert filter == {"product": "Oracle VPN"}
            document = SimpleNamespace(page_content="VPN connection troubleshooting", metadata={"document_id": "vpn-guide", "product": "Oracle VPN"})
            return [(document, 0.3)]

    backend = Backend()
    store = OracleVSVectorStore(backend)
    document = KnowledgeDocument(id="vpn-guide", content="VPN connection troubleshooting", source="official-guide", product="Oracle VPN")
    store.upsert([document], [[0.4, 0.6]])
    assert backend.inserted["ids"] == ["vpn-guide"]
    results = store.similarity_search([0.4, 0.6], k=2, filters=RetrievalFilters(product="Oracle VPN"))
    assert results[0].document_id == "vpn-guide"


def test_oraclevs_adapter_requires_expected_backend_capability() -> None:
    with pytest.raises(VectorStoreError, match="add_embeddings"):
        OracleVSVectorStore(object()).upsert([], [])
