"""Coordinate proactive support signals using shared contracts and injected evidence.

The service is deliberately side-effect-free: it reads a message and optional
ConversationState, then returns ProactiveAnalysis without changing conversation data.
Future integrations should inject reviewed providers instead of adding retrieval,
database, ticket-routing, or OCI SDK logic here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from src.models.conversation import ConversationState
from src.models.proactive import ArticleReference, ProactiveAnalysis, Sentiment

from .escalation import escalation_reason
from .interfaces import (
    HistoryProvider,
    RecommendationProvider,
    SentimentAnalyzer,
    UnsupportedIssueDetector,
)
from .recommendations import normalize_references
from .scoring import calculate_frustration_score
from .sentiment import RuleBasedSentimentAnalyzer

_LOGGER = logging.getLogger(__name__)


class ProactiveSupportService:
    """Produce proactive signals while isolating optional external dependencies."""

    def __init__(
        self,
        *,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        recommendation_provider: RecommendationProvider | None = None,
        history_provider: HistoryProvider | None = None,
        unsupported_issue_detector: UnsupportedIssueDetector | None = None,
    ) -> None:
        self._sentiment_analyzer = sentiment_analyzer or RuleBasedSentimentAnalyzer()
        self._recommendation_provider = recommendation_provider
        self._history_provider = history_provider
        self._unsupported_issue_detector = unsupported_issue_detector

    def analyze(
        self, message: str, conversation: ConversationState | None = None
    ) -> ProactiveAnalysis:
        """Analyze safely, treating unavailable provider evidence as absent."""
        if not isinstance(message, str) or not message.strip():
            return ProactiveAnalysis(sentiment=Sentiment.UNKNOWN)

        sentiment = self._safe_sentiment(message)
        frustration_score = calculate_frustration_score(message, sentiment, conversation)
        unsupported_issue = self._is_unsupported(message, conversation)
        reason = escalation_reason(
            message,
            frustration_score,
            conversation,
            unsupported_issue=unsupported_issue,
        )
        return ProactiveAnalysis(
            sentiment=sentiment,
            frustration_score=frustration_score,
            escalation_required=reason is not None,
            reason=reason,
            recommended_articles=self._recommendations(message, conversation),
        )

    def _safe_sentiment(self, message: str) -> Sentiment:
        try:
            result = self._sentiment_analyzer.analyze(message)
            return result if isinstance(result, Sentiment) else Sentiment(str(result).upper())
        except (Exception, ValueError):
            _LOGGER.warning("Proactive sentiment provider unavailable; using UNKNOWN sentiment.")
            return Sentiment.UNKNOWN

    def _is_unsupported(self, message: str, conversation: ConversationState | None) -> bool:
        if self._unsupported_issue_detector is None:
            return False
        try:
            return bool(self._unsupported_issue_detector.is_unsupported(message, conversation))
        except Exception:
            _LOGGER.warning("Proactive unsupported-issue provider unavailable.")
            return False

    def _recommendations(
        self, message: str, conversation: ConversationState | None
    ) -> list[ArticleReference]:
        groups: tuple[tuple[Any, str], ...] = (
            (self._call_provider(self._recommendation_provider, "related_articles", message, conversation), "knowledge_article"),
            (self._call_provider(self._recommendation_provider, "similar_issues", message, conversation), "similar_issue"),
            (self._call_provider(self._history_provider, "historical_solutions", message, conversation), "historical_solution"),
            (self._call_provider(self._history_provider, "customer_history", message, conversation), "customer_history"),
        )
        articles: list[ArticleReference] = []
        for evidence, source in groups:
            if isinstance(evidence, Iterable) and not isinstance(evidence, (str, bytes, dict)):
                articles.extend(normalize_references(evidence, source))
        return articles

    @staticmethod
    def _call_provider(
        provider: Any, method_name: str, message: str, conversation: ConversationState | None
    ) -> Any:
        if provider is None:
            return ()
        method: Callable[..., Any] | None = getattr(provider, method_name, None)
        if not callable(method):
            return ()
        try:
            return method(message, conversation)
        except Exception:
            _LOGGER.warning("Proactive evidence provider unavailable for %s.", method_name)
            return ()
