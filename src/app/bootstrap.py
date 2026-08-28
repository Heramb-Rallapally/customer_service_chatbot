"""Lazy, explicit runtime composition for the customer-support application.

Importing this module has no Ollama, OCI, or Oracle Database side effects. Calling
``create_application`` builds the production graph, unless callers inject
already-constructed services for tests or an alternate hosting environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import Settings, get_settings
from src.analytics import AnalyticsEventSink, InMemoryAnalyticsEventSink, NoOpAnalyticsEventSink
from src.conversation import (
    ConversationEngine,
    ConversationMemory,
    InMemoryConversationMemory,
    OracleConversationMemory,
)
from src.conversation.interfaces import LLMService, ProactiveService, Retriever
from src.ingestion import DocumentIndexer, KnowledgeIndexer
from src.llm import OciCohereLLMService, OllamaLLMService
from src.ollama import OllamaApiClient, OllamaClient, OllamaClientError
from src.proactive import (
    ConversationMemoryHistoryProvider,
    ProactiveSupportService,
    RetrievalEvidenceProvider,
)
from src.proactive.interfaces import (
    HistoryProvider,
    RecommendationProvider,
    SentimentAnalyzer,
    UnsupportedIssueDetector,
)
from src.retrieval import (
    OCIEmbeddingService,
    OllamaEmbeddingService,
    OracleVSVectorStore,
    RetrievalService,
    SimilarityMetric,
)
from src.retrieval.interfaces import EmbeddingService


class ApplicationConfigurationError(ValueError):
    """Raised for missing or invalid non-secret application configuration."""


class ApplicationInitializationError(RuntimeError):
    """Raised when an external production dependency cannot be initialized."""


@dataclass
class ApplicationServices:
    """Runtime service graph and its owned resource lifecycle.

    ``ORACLE_CONVERSATION_TABLE`` selects durable Oracle-backed memory.
    Without it, memory is deliberately process-local and suitable only for
    local development or a single application process.
    """

    settings: Settings
    conversation_engine: ConversationEngine
    retrieval_service: Retriever
    embeddings: Optional[EmbeddingService]
    vector_store: Optional[OracleVSVectorStore]
    llm_service: LLMService
    proactive_service: ProactiveService
    memory: ConversationMemory
    knowledge_indexer: KnowledgeIndexer
    analytics_sink: AnalyticsEventSink = field(default_factory=NoOpAnalyticsEventSink)
    oracle_connection: Any = None
    ollama_client: Optional[OllamaClient] = None
    _owns_oracle_connection: bool = field(default=False, repr=False)
    _owns_ollama_client: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close model-provider and Oracle resources owned by this container."""

        if self._closed:
            return
        self._closed = True
        if self._owns_ollama_client and self.ollama_client is not None:
            self.ollama_client.close()
        if self._owns_oracle_connection and self.oracle_connection is not None:
            close = getattr(self.oracle_connection, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "ApplicationServices":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_services(**kwargs: Any) -> ApplicationServices:
    """Compatibility-friendly name for building the complete service graph."""

    return create_application(**kwargs)


def create_application(
    *,
    settings: Optional[Settings] = None,
    oracle_connection: Any = None,
    embeddings: Optional[EmbeddingService] = None,
    oracle_backend: Any = None,
    retrieval_service: Optional[Retriever] = None,
    proactive_service: Optional[ProactiveService] = None,
    proactive_sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    proactive_recommendation_provider: Optional[RecommendationProvider] = None,
    proactive_history_provider: Optional[HistoryProvider] = None,
    proactive_unsupported_issue_detector: Optional[UnsupportedIssueDetector] = None,
    memory: Optional[ConversationMemory] = None,
    llm_service: Optional[LLMService] = None,
    analytics_sink: Optional[AnalyticsEventSink] = None,
    ollama_client: Optional[OllamaClient] = None,
) -> ApplicationServices:
    """Construct the real dependency graph or use explicitly injected doubles.

    Production callers normally provide no optional dependency arguments.  Unit
    tests can inject a complete ``retrieval_service`` and ``llm_service`` to
    avoid model-provider and Oracle initialization altogether.
    """

    configured_settings = settings or get_settings()
    resolved_analytics_sink = analytics_sink or _analytics_sink_from_settings(configured_settings)
    use_durable_memory = (
        memory is None and configured_settings.oracle_conversation_table is not None
    )
    _validate_configuration(
        configured_settings,
        needs_retrieval=retrieval_service is None,
        needs_oraclevs_backend=retrieval_service is None and oracle_backend is None,
        needs_oracle_connection=(
            (
                retrieval_service is None
                and oracle_backend is None
                and oracle_connection is None
            )
            or (use_durable_memory and oracle_connection is None)
        ),
        needs_embeddings=retrieval_service is None and embeddings is None,
        needs_llm=llm_service is None,
        needs_durable_memory=use_durable_memory,
    )

    owns_connection = False
    owns_ollama_client = False
    connection = oracle_connection
    resolved_ollama_client = ollama_client
    resolved_embeddings = embeddings
    resolved_store: Optional[OracleVSVectorStore] = None
    try:
        ollama_models: list[str] = []
        if (
            retrieval_service is None
            and resolved_embeddings is None
            and _provider_name(
                configured_settings.embedding_provider, "EMBEDDING_PROVIDER"
            ) == "ollama"
        ):
            ollama_models.append(
                configured_settings.embedding_model or "nomic-embed-text"
            )
        if (
            llm_service is None
            and _provider_name(configured_settings.llm_provider, "LLM_PROVIDER")
            == "ollama"
        ):
            ollama_models.append(configured_settings.llm_model or "llama3.2:3b")
        if ollama_models:
            if resolved_ollama_client is None:
                resolved_ollama_client = _create_ollama_client(configured_settings)
                owns_ollama_client = True
            _ensure_ollama_models(resolved_ollama_client, ollama_models)

        if connection is None and use_durable_memory:
            connection = _create_oracle_connection(configured_settings)
            owns_connection = True
        if retrieval_service is None:
            if resolved_embeddings is None:
                if _provider_name(
                    configured_settings.embedding_provider, "EMBEDDING_PROVIDER"
                ) == "ollama":
                    assert resolved_ollama_client is not None
                    resolved_embeddings = OllamaEmbeddingService(
                        resolved_ollama_client,
                        model_id=configured_settings.embedding_model or "nomic-embed-text",
                        embedding_dimension=configured_settings.embedding_dimension or 768,
                    )
                else:
                    resolved_embeddings = _create_oci_embeddings(configured_settings)
            if connection is None and oracle_backend is None:
                connection = _create_oracle_connection(configured_settings)
                owns_connection = True
            backend = oracle_backend or _create_oraclevs_backend(
                connection,
                resolved_embeddings,
                configured_settings,
                embedding_dimension=getattr(
                    resolved_embeddings,
                    "embedding_dimension",
                    configured_settings.embedding_dimension,
                ),
            )
            resolved_store = OracleVSVectorStore(
                backend,
                metric=SimilarityMetric.COSINE,
                embedding_dimension=getattr(
                    resolved_embeddings,
                    "embedding_dimension",
                    configured_settings.embedding_dimension,
                ),
            )
            resolved_retrieval: Retriever = RetrievalService(resolved_embeddings, resolved_store)
        else:
            resolved_retrieval = retrieval_service

        if llm_service is not None:
            resolved_llm = llm_service
        elif _provider_name(configured_settings.llm_provider, "LLM_PROVIDER") == "ollama":
            assert resolved_ollama_client is not None
            resolved_llm = OllamaLLMService(
                resolved_ollama_client,
                model_id=configured_settings.llm_model or "llama3.2:3b",
            )
        else:
            resolved_llm = _create_oci_llm(configured_settings)
        resolved_memory = memory or (
            OracleConversationMemory(
                connection,
                table_name=configured_settings.oracle_conversation_table,
            )
            if use_durable_memory
            else InMemoryConversationMemory()
        )
        if proactive_service is not None:
            resolved_proactive = proactive_service
        else:
            evidence_provider = (
                proactive_recommendation_provider
                or RetrievalEvidenceProvider(resolved_retrieval)
            )
            unsupported_detector = proactive_unsupported_issue_detector
            if unsupported_detector is None and callable(
                getattr(evidence_provider, "is_unsupported", None)
            ):
                unsupported_detector = evidence_provider
            resolved_proactive = ProactiveSupportService(
                sentiment_analyzer=proactive_sentiment_analyzer,
                recommendation_provider=evidence_provider,
                history_provider=(
                    proactive_history_provider
                    or ConversationMemoryHistoryProvider(resolved_memory)
                ),
                unsupported_issue_detector=(
                    unsupported_detector or RetrievalEvidenceProvider(resolved_retrieval)
                ),
            )
        if not isinstance(resolved_retrieval, DocumentIndexer):
            raise ApplicationInitializationError(
                "Configured retrieval service does not support document indexing"
            )
        knowledge_indexer = KnowledgeIndexer(resolved_retrieval)
        engine = ConversationEngine(
            retriever=resolved_retrieval,
            llm_service=resolved_llm,
            proactive_service=resolved_proactive,
            memory=resolved_memory,
        )
    except Exception:
        if owns_connection:
            _close_quietly(connection)
        if owns_ollama_client and resolved_ollama_client is not None:
            resolved_ollama_client.close()
        raise

    return ApplicationServices(
        settings=configured_settings,
        conversation_engine=engine,
        retrieval_service=resolved_retrieval,
        embeddings=resolved_embeddings,
        vector_store=resolved_store,
        llm_service=resolved_llm,
        proactive_service=resolved_proactive,
        memory=resolved_memory,
        knowledge_indexer=knowledge_indexer,
        analytics_sink=resolved_analytics_sink,
        oracle_connection=connection,
        ollama_client=resolved_ollama_client,
        _owns_oracle_connection=owns_connection,
        _owns_ollama_client=owns_ollama_client,
    )


def _analytics_sink_from_settings(settings: Settings) -> AnalyticsEventSink:
    mode = settings.analytics_mode.strip().lower()
    if mode == "noop":
        return NoOpAnalyticsEventSink()
    if mode == "memory":
        return InMemoryAnalyticsEventSink()
    raise ApplicationConfigurationError("ANALYTICS_MODE must be 'noop' or 'memory'")


def _validate_configuration(
    settings: Settings,
    *,
    needs_retrieval: bool,
    needs_oraclevs_backend: bool,
    needs_oracle_connection: bool,
    needs_embeddings: bool,
    needs_llm: bool,
    needs_durable_memory: bool,
) -> None:
    embedding_provider = _provider_name(settings.embedding_provider, "EMBEDDING_PROVIDER")
    llm_provider = _provider_name(settings.llm_provider, "LLM_PROVIDER")
    if needs_retrieval and needs_oraclevs_backend:
        _require_settings(settings, "oracle_vs_table")
    if needs_durable_memory:
        _require_settings(settings, "oracle_conversation_table")
    if needs_oracle_connection:
        _require_settings(
            settings,
            "oracle_db_user",
            "oracle_db_password",
            "oracle_db_dsn",
        )
    if needs_embeddings:
        _require_settings(settings, "embedding_model", "embedding_dimension")
        if settings.embedding_dimension is not None and settings.embedding_dimension < 1:
            raise ApplicationConfigurationError("EMBEDDING_DIMENSION must be positive")
        if embedding_provider == "oci":
            _require_settings(settings, "oci_compartment_id")
        elif settings.embedding_dimension != 768 and settings.embedding_model == "nomic-embed-text":
            raise ApplicationConfigurationError(
                "nomic-embed-text requires EMBEDDING_DIMENSION=768"
            )
    if needs_oraclevs_backend and not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_$#]{0,127}", settings.oracle_vs_table or ""
    ):
        raise ApplicationConfigurationError(
            "ORACLEVS_TABLE must be a single valid Oracle identifier"
        )
    if needs_durable_memory and not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_$#]{0,127}", settings.oracle_conversation_table or ""
    ):
        raise ApplicationConfigurationError(
            "ORACLE_CONVERSATION_TABLE must be a single valid Oracle identifier"
        )
    if needs_llm:
        _require_settings(settings, "llm_model")
        if llm_provider == "oci":
            _require_settings(settings, "oci_compartment_id")
    if (needs_embeddings and embedding_provider == "ollama") or (
        needs_llm and llm_provider == "ollama"
    ):
        if not settings.ollama_base_url.strip():
            raise ApplicationConfigurationError("OLLAMA_BASE_URL must not be blank")
        if settings.ollama_timeout_seconds <= 0:
            raise ApplicationConfigurationError("OLLAMA_TIMEOUT_SECONDS must be positive")


def _provider_name(value: str, environment_name: str) -> str:
    provider = value.strip().lower()
    if provider not in {"ollama", "oci"}:
        raise ApplicationConfigurationError(
            f"{environment_name} must be 'ollama' or 'oci'"
        )
    return provider


def _require_settings(settings: Settings, *attributes: str) -> None:
    missing = [attribute.upper() for attribute in attributes if getattr(settings, attribute) is None]
    if missing:
        environment_names = {
            "ORACLE_DB_USER": "ORACLE_DB_USER",
            "ORACLE_DB_PASSWORD": "ORACLE_DB_PASSWORD",
            "ORACLE_DB_DSN": "ORACLE_DB_DSN",
            "ORACLE_VS_TABLE": "ORACLEVS_TABLE",
            "ORACLE_CONVERSATION_TABLE": "ORACLE_CONVERSATION_TABLE",
            "OCI_COMPARTMENT_ID": "OCI_COMPARTMENT_ID",
            "EMBEDDING_MODEL": "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSION": "EMBEDDING_DIMENSION",
            "LLM_MODEL": "LLM_MODEL",
        }
        names = [environment_names[name] for name in missing]
        raise ApplicationConfigurationError("Missing required configuration: " + ", ".join(names))


def _create_oracle_connection(settings: Settings) -> Any:
    """Create the configured Oracle client only during application construction."""

    try:
        import oracledb
    except ImportError as exc:  # pragma: no cover - dependency is pinned in production.
        raise ApplicationInitializationError("The Oracle Database driver is not installed") from exc
    try:
        return oracledb.connect(
            user=settings.oracle_db_user,
            password=settings.oracle_db_password.get_secret_value()
            if settings.oracle_db_password is not None
            else None,
            dsn=settings.oracle_db_dsn,
        )
    except Exception as exc:
        raise ApplicationInitializationError("Oracle Database connection failed") from exc


def _create_oraclevs_backend(
    connection: Any,
    embeddings: EmbeddingService,
    settings: Settings,
    *,
    embedding_dimension: Optional[int],
) -> Any:
    """Build the exact pinned LangChain OracleVS backend with COSINE distance."""

    try:
        from langchain_community.vectorstores.oraclevs import OracleVS
        from langchain_community.vectorstores.utils import DistanceStrategy
    except ImportError as exc:  # pragma: no cover - dependencies are pinned in production.
        raise ApplicationInitializationError("The pinned LangChain OracleVS dependency is not installed") from exc
    if embedding_dimension is not None:
        _validate_oracle_vector_dimension(
            connection,
            table_name=settings.oracle_vs_table or "",
            expected_dimension=embedding_dimension,
        )
    try:
        return OracleVS(
            connection,
            embeddings,
            settings.oracle_vs_table,
            distance_strategy=DistanceStrategy.COSINE,
        )
    except Exception as exc:
        raise ApplicationInitializationError("OracleVS initialization failed") from exc


def _validate_oracle_vector_dimension(
    connection: Any, *, table_name: str, expected_dimension: int
) -> None:
    """Reject an existing OracleVS table created for another embedding model."""

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name = :table_name",
                table_name=table_name.upper(),
            )
            if int(cursor.fetchone()[0]) == 0:
                return
            cursor.execute(
                "SELECT DBMS_METADATA.GET_DDL('TABLE', :table_name) FROM dual",
                table_name=table_name.upper(),
            )
            raw_ddl = cursor.fetchone()[0]
            read = getattr(raw_ddl, "read", None)
            ddl = read() if callable(read) else raw_ddl
    except Exception as exc:
        raise ApplicationInitializationError(
            "Oracle vector schema dimension validation failed"
        ) from exc
    if not isinstance(ddl, str):
        raise ApplicationConfigurationError(
            "Existing OracleVS table metadata is unavailable"
        )
    match = re.search(
        r'"?EMBEDDING"?\s+VECTOR\(\s*(\d+)\s*,', ddl, flags=re.IGNORECASE
    )
    if match is None:
        raise ApplicationConfigurationError(
            "Existing OracleVS table does not expose a fixed embedding dimension"
        )
    stored_dimension = int(match.group(1))
    if stored_dimension != expected_dimension:
        raise ApplicationConfigurationError(
            "Existing OracleVS table embedding dimension does not match the configured model; "
            "use a correctly provisioned table and re-index the knowledge base"
        )


def _create_ollama_client(settings: Settings) -> OllamaApiClient:
    return OllamaApiClient(
        settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
    )


def _ensure_ollama_models(client: OllamaClient, models: list[str]) -> None:
    try:
        client.ensure_models(models)
    except OllamaClientError as exc:
        raise ApplicationInitializationError(
            "Ollama is unavailable or a configured model is not installed"
        ) from exc


def _create_oci_embeddings(settings: Settings) -> OCIEmbeddingService:
    try:
        return OCIEmbeddingService.from_settings(settings)
    except Exception as exc:
        raise ApplicationInitializationError("OCI embedding initialization failed") from exc


def _create_oci_llm(settings: Settings) -> OciCohereLLMService:
    try:
        return OciCohereLLMService.from_settings(settings)
    except Exception as exc:
        raise ApplicationInitializationError("OCI LLM initialization failed") from exc


def _close_quietly(connection: Any) -> None:
    close = getattr(connection, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
