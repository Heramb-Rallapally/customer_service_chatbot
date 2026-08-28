"""Unit tests for Member 2 retrieval components; no OCI or Oracle required."""

import inspect
from collections.abc import Sequence
from importlib.metadata import version
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from src.config import Settings
from src.models import KnowledgeDocument, RetrievalResult, Severity
from src.retrieval import (
    HashEmbeddingService,
    InMemoryVectorStore,
    OCIEmbeddingService,
    OracleScoreSemantics,
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
    results = service.search(query="VPN connection", filters={}, top_k=1)
    assert [result.document_id for result in results] == ["vpn-guide"]
    assert results[0].score > 0


def test_search_exposes_conversation_facing_signature(service: RetrievalService) -> None:
    signature = inspect.signature(service.search)
    assert list(signature.parameters) == ["query", "filters", "top_k"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
    assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())
    assert not inspect.iscoroutinefunction(service.search)
    assert get_type_hints(service.search)["return"] == Sequence[RetrievalResult]
    assert [result.document_id for result in service.search(query="VPN", filters={}, top_k=1)] == ["vpn-guide"]
    with pytest.raises(ValueError, match="k must be at least 1"):
        service.search(query="VPN", filters={}, top_k=0)


def test_search_accepts_member_3_mapping_filters(service: RetrievalService) -> None:
    results = service.search(query="connection", filters={"product": "Oracle Mail"}, top_k=1)
    assert [result.document_id for result in results] == ["email-guide"]


def test_search_normalizes_conversation_filters_and_rejects_unknown_keys(service: RetrievalService) -> None:
    results = service.search(query="connection", filters={"product": " oracle vpn "}, top_k=1)
    assert [result.document_id for result in results] == ["vpn-guide"]
    with pytest.raises(ValueError, match="unsupported"):
        service.search(query="connection", filters={"source": "official-guide"}, top_k=1)


@pytest.mark.parametrize(
    "filters",
    [
        {"product": "oracle vpn"},
        {"version": " 5.2 "},
        {"issue_type": "connectivity"},
        {"severity": "medium"},
        {
            "product": "ORACLE VPN",
            "version": "5.2",
            "issue_type": "CONNECTIVITY",
            "severity": "MEDIUM",
        },
    ],
)
def test_conversation_filter_keys_match_normalized_metadata(filters: dict[str, str]) -> None:
    document = KnowledgeDocument(
        id="vpn",
        content="vpn connectivity",
        source="guide",
        product="Oracle VPN",
        version="5.2",
        issue_type="Connectivity",
        severity=Severity.MEDIUM,
    )
    service = RetrievalService(HashEmbeddingService(), InMemoryVectorStore())
    service.index_documents([document])
    results = service.search(query="vpn", filters=filters, top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)


@pytest.mark.parametrize("metric", list(SimilarityMetric))
def test_in_memory_scores_are_normalized_and_higher_is_better(metric: SimilarityMetric) -> None:
    store = InMemoryVectorStore(metric)
    documents = [
        KnowledgeDocument(id="match", content="match", source="test"),
        KnowledgeDocument(id="other", content="other", source="test"),
    ]
    store.upsert(documents, [[1.0, 0.0], [0.0, 1.0]])
    results = store.similarity_search([1.0, 0.0], k=2, filters=None)
    assert [result.document_id for result in results] == ["match", "other"]
    assert all(0.0 <= result.score <= 1.0 for result in results)


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
    assert adapter.embedding_dimension == 2


def test_oci_embedding_adapter_rejects_dimension_mismatch() -> None:
    class Client:
        def embed_text(self, _request):
            return SimpleNamespace(data=SimpleNamespace(embeddings=[[0.1, 0.2]]))

    adapter = OCIEmbeddingService(
        Client(),
        compartment_id="compartment",
        model_id="model",
        request_factory=list,
        embedding_dimension=3,
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        adapter.embed_query("vpn")


def test_oci_adapter_requires_configuration() -> None:
    with pytest.raises(EmbeddingError, match="OCI_COMPARTMENT_ID"):
        OCIEmbeddingService.from_settings(Settings())


def test_oraclevs_adapter_maps_insert_and_search() -> None:
    class Backend:
        def add_texts(self, *, texts, metadatas, ids):
            self.inserted = {"texts": texts, "metadatas": metadatas, "ids": ids}

        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            assert embedding == [0.4, 0.6]
            assert k == 2
            document = SimpleNamespace(page_content="VPN connection troubleshooting", metadata={"document_id": "vpn-guide", "product": "Oracle VPN", "source": "official-guide"})
            return [(document, 0.3)]

    backend = Backend()
    store = OracleVSVectorStore(backend, score_semantics=OracleScoreSemantics.DISTANCE)
    document = KnowledgeDocument(id="vpn-guide", content="VPN connection troubleshooting", source="official-guide", product="Oracle VPN")
    store.upsert([document], [[0.4, 0.6]])
    assert backend.inserted["ids"] == ["vpn-guide"]
    assert backend.inserted["texts"] == ["VPN connection troubleshooting"]
    assert backend.inserted["metadatas"][0]["source"] == "official-guide"
    results = store.similarity_search([0.4, 0.6], k=2, filters=RetrievalFilters(product="Oracle VPN"))
    assert results[0].document_id == "vpn-guide"
    assert results[0].score == pytest.approx(0.85)


def test_retrieval_uses_one_oci_embedding_batch_for_oraclevs_text_indexing() -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def embed_text(self, request):
            self.requests.append(request)
            return SimpleNamespace(data=SimpleNamespace(embeddings=[[0.4, 0.6] for _ in request]))

    class Backend:
        def __init__(self, embedding_function) -> None:
            self.embedding_function = embedding_function
            self.inserted = None

        def add_texts(self, *, texts, metadatas, ids):
            # This matches the pinned OracleVS add_texts behavior: it owns
            # document embedding before inserting rows.
            vectors = self.embedding_function.embed_documents(texts)
            self.inserted = (texts, metadatas, ids, vectors)

    client = Client()
    embeddings = OCIEmbeddingService(
        client,
        compartment_id="compartment",
        model_id="model",
        request_factory=list,
        embedding_dimension=2,
    )
    backend = Backend(embeddings)
    service = RetrievalService(
        embeddings,
        OracleVSVectorStore(backend, embedding_dimension=2),
    )
    documents = [
        KnowledgeDocument(
            id="vpn-guide",
            content="VPN connection troubleshooting",
            source="official-guide",
            product="Oracle VPN",
            version="5.2",
            severity=Severity.MEDIUM,
            metadata={"authoritative": True},
        ),
        KnowledgeDocument(
            id="mail-guide",
            content="Reset the mailbox password",
            source="official-guide",
            product="Oracle Mail",
        ),
    ]

    service.index_documents(documents)

    assert client.requests == [[document.content for document in documents]]
    assert backend.inserted[2] == ["vpn-guide", "mail-guide"]
    assert backend.inserted[0] == [document.content for document in documents]
    assert backend.inserted[1][0]["product"] == "Oracle VPN"
    assert backend.inserted[1][0]["document_id"] == "vpn-guide"
    assert backend.inserted[1][0]["authoritative"] is True


def test_oraclevs_rejects_non_json_metadata_before_insertion() -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def add_texts(self, **_kwargs):
            self.calls += 1

    backend = Backend()
    document = KnowledgeDocument(
        id="invalid-metadata",
        content="content",
        source="source",
        metadata={"unsupported": object()},
    )

    with pytest.raises(VectorStoreError, match="JSON serializable"):
        OracleVSVectorStore(backend).index_documents([document])

    assert backend.calls == 0


def test_oraclevs_text_indexing_keeps_oci_dimension_validation() -> None:
    class Client:
        def embed_text(self, _request):
            return SimpleNamespace(data=SimpleNamespace(embeddings=[[0.1]]))

    class Backend:
        def __init__(self, embedding_function) -> None:
            self.embedding_function = embedding_function
            self.inserted = False

        def add_texts(self, *, texts, **_kwargs):
            self.embedding_function.embed_documents(texts)
            self.inserted = True

    embeddings = OCIEmbeddingService(
        Client(),
        compartment_id="compartment",
        model_id="model",
        request_factory=list,
        embedding_dimension=2,
    )
    backend = Backend(embeddings)
    service = RetrievalService(embeddings, OracleVSVectorStore(backend, embedding_dimension=2))

    with pytest.raises(VectorStoreError, match="document insertion failed") as error:
        service.index_documents(
            [KnowledgeDocument(id="dimension", content="content", source="source")]
        )

    assert isinstance(error.value.__cause__, EmbeddingError)
    assert backend.inserted is False


def test_oraclevs_insert_only_reindexing_preserves_duplicate_failure() -> None:
    duplicate_error = RuntimeError("unique constraint violated")

    class Backend:
        def __init__(self) -> None:
            self.ids: set[str] = set()

        def add_texts(self, *, ids, **_kwargs):
            if any(identifier in self.ids for identifier in ids):
                raise duplicate_error
            self.ids.update(ids)

    store = OracleVSVectorStore(Backend())
    document = KnowledgeDocument(id="stable-id", content="content", source="source")
    store.index_documents([document])

    with pytest.raises(VectorStoreError, match="document insertion failed") as error:
        store.index_documents([document])

    assert error.value.__cause__ is duplicate_error


def test_oraclevs_progressively_overfetches_before_local_filtering() -> None:
    non_matching = SimpleNamespace(page_content="Mail guide", metadata={"document_id": "mail", "product": "Oracle Mail", "source": "official-guide"})
    matching = SimpleNamespace(page_content="VPN guide", metadata={"document_id": "vpn", "product": "Oracle VPN", "source": "official-guide"})

    class Backend:
        calls: list[int] = []

        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            assert embedding == [0.4, 0.6]
            self.calls.append(k)
            return [(non_matching, 0.1), (matching, 0.2)][:k]

    backend = Backend()
    results = OracleVSVectorStore(backend).similarity_search(
        [0.4, 0.6], k=1, filters=RetrievalFilters(product="Oracle VPN")
    )
    assert [result.document_id for result in results] == ["vpn"]
    assert backend.calls == [1, 2]


@pytest.mark.parametrize(
    ("metric", "raw_distances"),
    [
        (SimilarityMetric.COSINE, [0.2, 1.2]),
        (SimilarityMetric.EUCLIDEAN, [0.1, 3.0]),
        (SimilarityMetric.DOT, [-2.0, 1.0]),
    ],
)
def test_oraclevs_distance_scores_are_normalized_and_descending(
    metric: SimilarityMetric, raw_distances: list[float]
) -> None:
    documents = [
        SimpleNamespace(page_content="First", metadata={"document_id": "first", "source": "official-guide"}),
        SimpleNamespace(page_content="Second", metadata={"document_id": "second", "source": "official-guide"}),
    ]

    class Backend:
        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            return list(zip(documents, raw_distances))

    results = OracleVSVectorStore(
        Backend(), metric=metric, score_semantics=OracleScoreSemantics.DISTANCE
    ).similarity_search([0.4, 0.6], k=2, filters=None)
    assert [result.document_id for result in results] == ["first", "second"]
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results[0].score > results[1].score
    assert all(result.metadata["source"] == "official-guide" for result in results)


def test_oraclevs_adapter_requires_expected_backend_capability() -> None:
    document = KnowledgeDocument(id="id", content="content", source="source")
    with pytest.raises(VectorStoreError, match="add_texts"):
        OracleVSVectorStore(
            object(), score_semantics=OracleScoreSemantics.DISTANCE
        ).upsert([document], [[0.1]])


def test_oraclevs_rejects_duplicate_batches_and_dimension_mismatches() -> None:
    class Backend:
        def add_texts(self, **_kwargs):
            pass

    store = OracleVSVectorStore(Backend(), embedding_dimension=2)
    document = KnowledgeDocument(id="id", content="content", source="source")
    with pytest.raises(VectorStoreError, match="dimension"):
        store.upsert([document], [[0.1]])
    with pytest.raises(VectorStoreError, match="duplicate"):
        store.upsert([document, document], [[0.1, 0.2], [0.1, 0.2]])


def test_oraclevs_rejects_non_finite_distances_and_preserves_missing_source() -> None:
    document = SimpleNamespace(page_content="content", metadata={"document_id": "id"})

    class Backend:
        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            return [(document, float("nan"))]

    with pytest.raises(VectorStoreError, match="invalid vector distance"):
        OracleVSVectorStore(Backend()).similarity_search([0.1], k=1, filters=None)

    class SourceLessBackend:
        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            return [(document, 0.0)]

    result = OracleVSVectorStore(SourceLessBackend()).similarity_search([0.1], k=1, filters=None)
    assert result[0].metadata == {"document_id": "id"}


def test_oraclevs_empty_batch_is_a_noop_and_cap_fails_visibly() -> None:
    class Backend:
        calls = 0

        def add_texts(self, **_kwargs):
            self.calls += 1

        def similarity_search_by_vector_with_relevance_scores(self, *, embedding, k):
            return [
                (SimpleNamespace(page_content="mail", metadata={"document_id": "mail", "product": "Mail"}), 0.0)
            ]

    backend = Backend()
    store = OracleVSVectorStore(backend, max_candidate_fetch=1)
    store.upsert([], [])
    assert backend.calls == 0
    with pytest.raises(VectorStoreError, match="max_candidate_fetch"):
        store.similarity_search([0.1], k=1, filters=RetrievalFilters(product="VPN"))


def test_pinned_oraclevs_surface_is_available() -> None:
    from langchain_community.vectorstores.oraclevs import OracleVS

    assert version("langchain-community") == "0.3.31"
    assert hasattr(OracleVS, "add_texts")
    assert hasattr(OracleVS, "similarity_search_by_vector_with_relevance_scores")
    assert not hasattr(OracleVS, "add_embeddings")
    assert not hasattr(OracleVS, "similarity_search_with_score_by_vector")
