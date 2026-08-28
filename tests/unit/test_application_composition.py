"""Credential-free tests for the lazy application composition root."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from src.app import (
    create_application,
    create_services,
)
from src.config import Settings
from src.conversation import GeneratedResponse, InMemoryConversationMemory
from src.llm import OciCohereLLMService
from src.proactive import ProactiveSupportService
from src.retrieval import OCIEmbeddingService, OracleVSVectorStore, RetrievalService


class FakeRetriever:
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
        bootstrap, "_create_oraclevs_backend", lambda *_args: Backend()
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
    assert isinstance(services.memory, InMemoryConversationMemory)
    assert services.conversation_engine._retriever is services.retrieval_service
    assert services.conversation_engine._llm_service is llm
    assert services.conversation_engine._proactive_service is services.proactive_service
    assert services.conversation_engine._memory is services.memory

    services.close()
    services.close()
    assert connection.close_calls == 1


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
