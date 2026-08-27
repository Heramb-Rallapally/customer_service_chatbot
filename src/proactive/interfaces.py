"""Define injectable proactive-support boundaries without owning external services.

This module keeps sentiment, recommendation, and history integrations replaceable so
the proactive workflow can be tested without OCI, retrieval, or database access.
Future engineers should extend these small protocols instead of coupling this package
to provider SDKs or persistence implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.models.conversation import ConversationState
from src.models.proactive import ArticleReference, Sentiment


class SentimentAnalyzer(Protocol):
    """Classify one customer message into the shared normalized sentiment enum."""

    def analyze(self, message: str) -> Sentiment:
        """Return a normalized sentiment without mutating conversation state."""


class RecommendationProvider(Protocol):
    """Supply evidence-backed related knowledge and similar-issue references."""

    def related_articles(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        """Return related knowledge articles supported by the provider's evidence."""

    def similar_issues(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        """Return references for similar issues supported by provider evidence."""


class HistoryProvider(Protocol):
    """Supply evidence-backed historical solution and customer-history references."""

    def historical_solutions(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        """Return references to relevant historical resolutions."""

    def customer_history(
        self, message: str, conversation: ConversationState | None
    ) -> Sequence[ArticleReference]:
        """Return references to relevant customer-history signals."""


class UnsupportedIssueDetector(Protocol):
    """Identify whether available evidence indicates the issue is unsupported."""

    def is_unsupported(
        self, message: str, conversation: ConversationState | None
    ) -> bool:
        """Return true only when the injected detector has such an unsupported signal."""
