"""Lazy, explicit runtime composition for the customer-support application.

Importing this module has no OCI or Oracle Database side effects.  Calling
``create_application`` builds the production graph, unless callers inject
already-constructed services for tests or an alternate hosting environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.config import Settings, get_settings
from src.conversation import (
    ConversationEngine,
    ConversationMemory,
    InMemoryConversationMemory,
    OracleConversationMemory,
)
from src.conversation.interfaces import LLMService, ProactiveService, Retriever
from src.ingestion import DocumentIndexer, KnowledgeIndexer
from src.llm import OciCohereLLMService
from src.proactive import ProactiveSupportService
from src.retrieval import (
    OCIEmbeddingService,
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
    oracle_connection: Any = None
    _owns_oracle_connection: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close the Oracle connection created by this container, if any."""

        if self._closed:
            return
        self._closed = True
        if not self._owns_oracle_connection or self.oracle_connection is None:
            return
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
    memory: Optional[ConversationMemory] = None,
    llm_service: Optional[LLMService] = None,
) -> ApplicationServices:
    """Construct the real dependency graph or use explicitly injected doubles.

    Production callers normally provide no optional dependency arguments.  Unit
    tests can inject a complete ``retrieval_service`` and ``llm_service`` to
    avoid OCI and Oracle initialization altogether.
    """

    configured_settings = settings or get_settings()
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
    connection = oracle_connection
    resolved_embeddings = embeddings
    resolved_store: Optional[OracleVSVectorStore] = None
    try:
        if connection is None and use_durable_memory:
            connection = _create_oracle_connection(configured_settings)
            owns_connection = True
        if retrieval_service is None:
            if resolved_embeddings is None:
                resolved_embeddings = OCIEmbeddingService.from_settings(configured_settings)
            if connection is None and oracle_backend is None:
                connection = _create_oracle_connection(configured_settings)
                owns_connection = True
            backend = oracle_backend or _create_oraclevs_backend(
                connection, resolved_embeddings, configured_settings
            )
            resolved_store = OracleVSVectorStore(
                backend,
                metric=SimilarityMetric.COSINE,
                embedding_dimension=configured_settings.embedding_dimension,
            )
            resolved_retrieval: Retriever = RetrievalService(resolved_embeddings, resolved_store)
        else:
            resolved_retrieval = retrieval_service

        resolved_llm = llm_service or OciCohereLLMService.from_settings(configured_settings)
        resolved_proactive = proactive_service or ProactiveSupportService()
        resolved_memory = memory or (
            OracleConversationMemory(
                connection,
                table_name=configured_settings.oracle_conversation_table,
            )
            if use_durable_memory
            else InMemoryConversationMemory()
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
        oracle_connection=connection,
        _owns_oracle_connection=owns_connection,
    )


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
        _require_settings(
            settings,
            "oci_compartment_id",
            "embedding_model",
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
        _require_settings(settings, "oci_compartment_id", "llm_model")


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
    connection: Any, embeddings: EmbeddingService, settings: Settings
) -> Any:
    """Build the exact pinned LangChain OracleVS backend with COSINE distance."""

    try:
        from langchain_community.vectorstores.oraclevs import OracleVS
        from langchain_community.vectorstores.utils import DistanceStrategy
    except ImportError as exc:  # pragma: no cover - dependencies are pinned in production.
        raise ApplicationInitializationError("The pinned LangChain OracleVS dependency is not installed") from exc
    try:
        return OracleVS(
            connection,
            embeddings,
            settings.oracle_vs_table,
            distance_strategy=DistanceStrategy.COSINE,
        )
    except Exception as exc:
        raise ApplicationInitializationError("OracleVS initialization failed") from exc


def _close_quietly(connection: Any) -> None:
    close = getattr(connection, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
