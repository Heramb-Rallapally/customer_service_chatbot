"""Credential-free tests for the lazy application composition root."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import httpx
import pytest

from src.app import (
    create_application,
    create_services,
)
from src.config import Settings
from src.conversation import GeneratedResponse, InMemoryConversationMemory, OracleConversationMemory
from src.llm import OciCohereLLMService, OllamaLLMService
from src.ollama import OllamaApiClient
from src.proactive import (
    ConversationMemoryHistoryProvider,
    ProactiveSupportService,
    RetrievalEvidenceProvider,
)
from src.retrieval import (
    OCIEmbeddingService,
    OllamaEmbeddingService,
    OracleVSVectorStore,
    RetrievalService,
)
from src.ingestion import KnowledgeIndexer
from src.analytics import InMemoryAnalyticsEventSink, NoOpAnalyticsEventSink


class FakeRetriever:
    def index_documents(self, _documents: object) -> None:
        pass

    def search(self, *, query: str, filters: dict[str, str], top_k: int) -> list[object]:
        return []


class FakeLLM:
    def generate(self, _context: object) -> GeneratedResponse:
        return GeneratedResponse(message="Test response")


class FakeConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class Backend:
    def __init__(self) -> None:
        from langchain_community.vectorstores.utils import DistanceStrategy

        self.distance_strategy = DistanceStrategy.COSINE


def production_settings() -> Settings:
    return Settings(
        llm_provider="oci",
        embedding_provider="oci",
        oci_compartment_id="compartment",
        embedding_model="cohere.embed-english-v3.0",
        embedding_dimension=3,
        llm_model="cohere.command-r-plus",
        oracle_db_user="app_user",
        oracle_db_password="not-a-real-secret",
        oracle_db_dsn="db.example.invalid/service",
        oracle_vs_table="SUPPORT_KNOWLEDGE",
    )


def test_import_is_credential_free_and_does_not_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    import oracledb

    def connection_attempt(**_kwargs: object) -> object:
        raise AssertionError("Oracle connection must not occur during import")

    monkeypatch.setattr(oracledb, "connect", connection_attempt)
    monkeypatch.setattr(httpx, "Client", connection_attempt)
    module = importlib.import_module("src.app")
    importlib.reload(importlib.import_module("src.app.bootstrap"))

    assert hasattr(module, "create_application")


def test_fully_injected_services_skip_external_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.app.bootstrap as bootstrap

    def unexpected_external_setup(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("external setup must not run for injected dependencies")

    monkeypatch.setattr(bootstrap, "_create_oracle_connection", unexpected_external_setup)
    monkeypatch.setattr(bootstrap.OciCohereLLMService, "from_settings", unexpected_external_setup)

    retriever = FakeRetriever()
    llm = FakeLLM()
    proactive = ProactiveSupportService()
    memory = InMemoryConversationMemory()
    services = create_application(
        settings=Settings(),
        retrieval_service=retriever,
        llm_service=llm,
        proactive_service=proactive,
        memory=memory,
    )

    assert services.retrieval_service is retriever
    assert services.llm_service is llm
    assert services.proactive_service is proactive
    assert services.memory is memory
    assert services.conversation_engine._retriever is retriever
    assert services.conversation_engine._llm_service is llm
    assert services.conversation_engine._proactive_service is proactive
    assert services.conversation_engine._memory is memory
    assert isinstance(services.analytics_sink, NoOpAnalyticsEventSink)


def test_analytics_sink_is_explicitly_injectable_or_enabled_for_local_memory() -> None:
    sink = InMemoryAnalyticsEventSink()
    injected = create_application(
        settings=Settings(),
        retrieval_service=FakeRetriever(),
        llm_service=FakeLLM(),
        analytics_sink=sink,
    )
    assert injected.analytics_sink is sink

    local = create_application(
        settings=Settings(analytics_mode="memory"),
        retrieval_service=FakeRetriever(),
        llm_service=FakeLLM(),
    )
    assert isinstance(local.analytics_sink, InMemoryAnalyticsEventSink)


def test_invalid_analytics_mode_is_a_safe_configuration_error() -> None:
    import src.app.bootstrap as bootstrap

    with pytest.raises(bootstrap.ApplicationConfigurationError, match="ANALYTICS_MODE"):
        create_application(
            settings=Settings(analytics_mode="external"),
            retrieval_service=FakeRetriever(),
            llm_service=FakeLLM(),
        )


def test_individual_proactive_providers_are_injectable_without_external_setup() -> None:
    class FixedSentiment:
        def analyze(self, _message: str):
            from src.models import Sentiment

            return Sentiment.NEUTRAL

    class Provider:
        def related_articles(self, *_args: object):
            return ()

        def similar_issues(self, *_args: object):
            return ()

        def is_unsupported(self, *_args: object) -> bool:
            return False

    class History:
        def historical_solutions(self, *_args: object):
            return ()

        def customer_history(self, *_args: object):
            return ()

    recommendation = Provider()
    history = History()
    sentiment = FixedSentiment()
    services = create_application(
        settings=Settings(),
        retrieval_service=FakeRetriever(),
        llm_service=FakeLLM(),
        proactive_sentiment_analyzer=sentiment,
        proactive_recommendation_provider=recommendation,
        proactive_history_provider=history,
        proactive_unsupported_issue_detector=recommendation,
    )

    proactive = services.proactive_service
    assert isinstance(proactive, ProactiveSupportService)
    assert proactive._sentiment_analyzer is sentiment
    assert proactive._recommendation_provider is recommendation
    assert proactive._history_provider is history
    assert proactive._unsupported_issue_detector is recommendation


def test_production_factory_builds_expected_concrete_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.app.bootstrap as bootstrap

    connection = FakeConnection()
    embeddings = OCIEmbeddingService(
        SimpleNamespace(),
        compartment_id="compartment",
        model_id="cohere.embed-english-v3.0",
        request_factory=lambda texts: texts,
        embedding_dimension=3,
    )
    llm = OciCohereLLMService(
        SimpleNamespace(),
        compartment_id="compartment",
        model_id="cohere.command-r-plus",
        request_factory=lambda prompt: prompt,
    )
    monkeypatch.setattr(bootstrap, "_create_oracle_connection", lambda _settings: connection)
    monkeypatch.setattr(
        bootstrap, "_create_oraclevs_backend", lambda *_args, **_kwargs: Backend()
    )
    monkeypatch.setattr(
        bootstrap.OCIEmbeddingService, "from_settings", lambda _settings: embeddings
    )
    monkeypatch.setattr(
        bootstrap.OciCohereLLMService, "from_settings", lambda _settings: llm
    )

    services = create_services(settings=production_settings())

    assert isinstance(services.retrieval_service, RetrievalService)
    assert services.embeddings is embeddings
    assert isinstance(services.vector_store, OracleVSVectorStore)
    assert services.llm_service is llm
    assert isinstance(services.llm_service, OciCohereLLMService)
    assert isinstance(services.proactive_service, ProactiveSupportService)
    assert isinstance(services.proactive_service._recommendation_provider, RetrievalEvidenceProvider)
    assert isinstance(services.proactive_service._history_provider, ConversationMemoryHistoryProvider)
    assert services.proactive_service._unsupported_issue_detector is services.proactive_service._recommendation_provider
    assert isinstance(services.memory, InMemoryConversationMemory)
    assert isinstance(services.knowledge_indexer, KnowledgeIndexer)
    assert services.conversation_engine._retriever is services.retrieval_service
    assert services.conversation_engine._llm_service is llm
    assert services.conversation_engine._proactive_service is services.proactive_service
    assert services.conversation_engine._memory is services.memory

    services.close()
    services.close()
    assert connection.close_calls == 1


def test_default_factory_selects_ollama_providers_with_one_model_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.app.bootstrap as bootstrap

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "llama3.2:3b"},
                ]
            },
        )

    http_client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    client = OllamaApiClient("http://ollama.test", client=http_client)
    monkeypatch.setattr(
        bootstrap,
        "_create_oci_embeddings",
        lambda _settings: pytest.fail("default graph must not initialize OCI embeddings"),
    )
    monkeypatch.setattr(
        bootstrap,
        "_create_oci_llm",
        lambda _settings: pytest.fail("default graph must not initialize OCI LLM"),
    )
    monkeypatch.setattr(
        bootstrap, "_create_oraclevs_backend", lambda *_args, **_kwargs: Backend()
    )

    services = create_application(
        settings=Settings(oracle_vs_table="SUPPORT_KNOWLEDGE"),
        oracle_connection=FakeConnection(),
        ollama_client=client,
    )

    assert isinstance(services.embeddings, OllamaEmbeddingService)
    assert isinstance(services.llm_service, OllamaLLMService)
    assert len(requests) == 1
    assert requests[0].url.path == "/api/tags"
    assert services.conversation_engine._llm_service is services.llm_service


def test_existing_oracle_vector_dimension_must_match_selected_model() -> None:
    import src.app.bootstrap as bootstrap

    class Cursor:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def execute(self, _statement: str, **parameters: object) -> None:
            assert parameters == {"table_name": "SUPPORT_KNOWLEDGE"}
            self.calls += 1

        def fetchone(self) -> tuple[object]:
            if self.calls == 1:
                return (1,)
            return ('CREATE TABLE SUPPORT_KNOWLEDGE (EMBEDDING VECTOR(1024, FLOAT32))',)

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    with pytest.raises(
        bootstrap.ApplicationConfigurationError, match="embedding dimension"
    ):
        bootstrap._validate_oracle_vector_dimension(
            Connection(), table_name="SUPPORT_KNOWLEDGE", expected_dimension=768
        )


def test_configuration_errors_are_safe_and_happen_before_external_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.app.bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap,
        "_create_oracle_connection",
        lambda _settings: pytest.fail("configuration must fail before connecting"),
    )

    with pytest.raises(bootstrap.ApplicationConfigurationError) as error:
        create_application(settings=Settings())

    assert "Missing required configuration" in str(error.value)
    assert "not-a-real-secret" not in str(error.value)


def test_invalid_vector_table_name_is_rejected_before_connection() -> None:
    import src.app.bootstrap as bootstrap

    settings = production_settings().model_copy(update={"oracle_vs_table": "bad table;drop"})

    with pytest.raises(bootstrap.ApplicationConfigurationError, match="valid Oracle identifier"):
        create_application(settings=settings, llm_service=FakeLLM())


def test_injected_connection_remains_caller_owned() -> None:
    connection = FakeConnection()
    services = create_application(
        settings=Settings(),
        oracle_connection=connection,
        retrieval_service=FakeRetriever(),
        llm_service=FakeLLM(),
    )

    services.close()

    assert connection.close_calls == 0


def test_oracle_conversation_memory_is_selected_when_configured() -> None:
    connection = FakeConnection()
    settings = Settings(oracle_conversation_table="CHAT_CONVERSATIONS")

    services = create_application(
        settings=settings,
        oracle_connection=connection,
        retrieval_service=FakeRetriever(),
        llm_service=FakeLLM(),
    )

    assert isinstance(services.memory, OracleConversationMemory)
    assert services.conversation_engine._memory is services.memory
    services.close()
    assert connection.close_calls == 0


def test_injected_oraclevs_backend_needs_no_database_configuration() -> None:
    embeddings = OCIEmbeddingService(
        SimpleNamespace(),
        compartment_id="compartment",
        model_id="embedding-model",
        request_factory=lambda texts: texts,
        embedding_dimension=3,
    )

    services = create_application(
        settings=Settings(),
        embeddings=embeddings,
        oracle_backend=Backend(),
        llm_service=FakeLLM(),
    )

    assert isinstance(services.retrieval_service, RetrievalService)
    assert services.oracle_connection is None


def test_injected_search_only_retriever_is_rejected_for_indexing() -> None:
    class SearchOnlyRetriever:
        def search(self, *, query: str, filters: dict[str, str], top_k: int) -> list[object]:
            return []

    import src.app.bootstrap as bootstrap

    with pytest.raises(bootstrap.ApplicationInitializationError, match="document indexing"):
        create_application(
            settings=Settings(),
            retrieval_service=SearchOnlyRetriever(),
            llm_service=FakeLLM(),
        )
