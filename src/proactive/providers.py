"""Production-capable proactive providers built on existing project ports."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from enum import Enum
from threading import RLock
from typing import Optional

from src.conversation.context import RetrievalQueryBuilder
from src.conversation.interfaces import ConversationMemory, Retriever
from src.models import ArticleReference, ConversationState, ResolutionStatus, RetrievalResult

from .interfaces import HistoryProvider, RecommendationProvider, UnsupportedIssueDetector

_LOGGER = logging.getLogger(__name__)


class SupportLevel(str, Enum):
    """Evidence interpretation kept internal to proactive providers."""

    SUPPORTED = "SUPPORTED"
    POTENTIALLY_SUPPORTED = "POTENTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNAVAILABLE = "UNAVAILABLE"


class RetrievalEvidenceProvider(RecommendationProvider, UnsupportedIssueDetector):
    """Reuse the existing retriever for related knowledge and support evidence.

    Results are cached by normalized query/filter context so the service's
    unsupported check plus related/similar recommendation calls perform one
    retrieval for a given proactive analysis context. The provider never
    creates embeddings or accesses OracleVS directly.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        top_k: int = 5,
        strong_score: float = 0.75,
        cache_size: int = 32,
    ) -> None:
        if top_k < 1 or not 0.0 <= strong_score <= 1.0 or cache_size < 1:
            raise ValueError("invalid retrieval evidence provider configuration")
        self._retriever = retriever
        self._top_k = top_k
        self._strong_score = strong_score
        self._cache_size = cache_size
        self._query_builder = RetrievalQueryBuilder()
        self._cache: OrderedDict[tuple[str, tuple[tuple[str, str], ...]], tuple[SupportLevel, tuple[RetrievalResult, ...]]] = OrderedDict()
        self._lock = RLock()

    def related_articles(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        _, results = self._evidence(message, conversation)
        return [reference for result in results if (reference := _article_reference(result))]

    def similar_issues(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        _, results = self._evidence(message, conversation)
        return [
            reference
            for result in results
            if _is_historical_issue(result) and (reference := _article_reference(result))
        ]

    def assess_support(
        self, message: str, conversation: ConversationState | None
    ) -> SupportLevel:
        return self._evidence(message, conversation)[0]

    def retrieval_results(
        self,
        message: str,
        conversation: ConversationState | None,
        *,
        top_k: int,
    ) -> Optional[Sequence[RetrievalResult]]:
        """Return cached evidence for the engine when its retrieval matches.

        ``None`` means evidence is unavailable, so the engine can attempt its
        own retrieval and preserve its established fallback behavior.
        """

        if top_k != self._top_k:
            return None
        level, results = self._evidence(message, conversation)
        return None if level is SupportLevel.UNAVAILABLE else results

    def is_unsupported(
        self, message: str, conversation: ConversationState | None
    ) -> bool:
        return self.assess_support(message, conversation) is SupportLevel.UNSUPPORTED

    def _evidence(
        self, message: str, conversation: ConversationState | None
    ) -> tuple[SupportLevel, tuple[RetrievalResult, ...]]:
        query, filters = self._query_and_filters(message, conversation)
        key = (query, tuple(sorted(filters.items())))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        try:
            results = tuple(
                result
                for result in self._retriever.search(query=query, filters=filters, top_k=self._top_k)
                if isinstance(result, RetrievalResult)
            )
            level = _support_level(results, self._strong_score)
        except Exception:
            _LOGGER.warning("Proactive retrieval evidence is unavailable.")
            level, results = SupportLevel.UNAVAILABLE, ()
        evidence = (level, results)
        with self._lock:
            self._cache[key] = evidence
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return evidence

    def _query_and_filters(
        self, message: str, conversation: ConversationState | None
    ) -> tuple[str, dict[str, str]]:
        normalized_message = " ".join(message.strip().split())
        if conversation is None:
            return normalized_message, {}
        query, filters = self._query_builder.build(conversation, normalized_message)
        return query, filters


class ConversationMemoryHistoryProvider(HistoryProvider):
    """Expose only the authenticated user's prior memory through references."""

    def __init__(self, memory: ConversationMemory, *, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self._memory = memory
        self._limit = limit

    def historical_solutions(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        return [
            _history_reference(state, "historical_solution")
            for state in self._relevant_history(conversation)
            if state.resolution_status is ResolutionStatus.RESOLVED
        ]

    def customer_history(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        return [
            _history_reference(state, "customer_history")
            for state in self._relevant_history(conversation)
        ]

    def _relevant_history(self, conversation: ConversationState | None) -> Sequence[ConversationState]:
        states = self._history(conversation)
        if conversation is None:
            return states
        return tuple(
            state
            for state in states
            if (conversation.product is None or state.product == conversation.product)
            and (conversation.issue_type is None or state.issue_type == conversation.issue_type)
        )

    def _history(self, conversation: ConversationState | None) -> Sequence[ConversationState]:
        if conversation is None or not conversation.user_id:
            return ()
        list_for_user = getattr(self._memory, "list_for_user", None)
        if not callable(list_for_user):
            return ()
        try:
            states = list_for_user(
                conversation.user_id,
                exclude_conversation_id=conversation.conversation_id,
                limit=self._limit,
            )
        except Exception:
            _LOGGER.warning("Proactive customer history is unavailable.")
            return ()
        return tuple(
            state
            for state in states
            if isinstance(state, ConversationState) and state.user_id == conversation.user_id
        )


def _support_level(results: Sequence[RetrievalResult], strong_score: float) -> SupportLevel:
    if not results:
        return SupportLevel.UNSUPPORTED
    if max(result.score for result in results) >= strong_score:
        return SupportLevel.SUPPORTED
    return SupportLevel.POTENTIALLY_SUPPORTED


def _article_reference(result: RetrievalResult) -> Optional[ArticleReference]:
    if not isinstance(result.document_id, str) or not result.document_id.strip():
        return None
    metadata: Mapping[str, object] = result.metadata if isinstance(result.metadata, Mapping) else {}
    title = metadata.get("title")
    source = metadata.get("source")
    return ArticleReference(
        article_id=result.document_id,
        title=title.strip() if isinstance(title, str) and title.strip() else None,
        source=source.strip() if isinstance(source, str) and source.strip() else None,
    )


def _is_historical_issue(result: RetrievalResult) -> bool:
    source_type = result.metadata.get("source_type") if isinstance(result.metadata, Mapping) else None
    return isinstance(source_type, str) and source_type.casefold() in {
        "historical_ticket",
        "support_ticket",
        "ticket",
    }


def _history_reference(state: ConversationState, source: str) -> ArticleReference:
    return ArticleReference(article_id=state.conversation_id, source=source)
