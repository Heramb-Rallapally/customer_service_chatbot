"""Provide deterministic sentiment analysis with an OCI-ready injected-client adapter.

The fallback supports local development and tests without credentials or network access.
The adapter deliberately imports no OCI SDK; production wiring must inject a reviewed
client and keep credentials outside this package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.models.proactive import Sentiment


class RuleBasedSentimentAnalyzer:
    """Classify common support language predictably for mock-first development."""

    _POSITIVE_TERMS = ("thank", "thanks", "great", "helpful", "worked", "resolved")
    _NEGATIVE_TERMS = (
        "angry",
        "awful",
        "broken",
        "disappointed",
        "frustrated",
        "hate",
        "not working",
        "terrible",
        "useless",
    )

    def analyze(self, message: str) -> Sentiment:
        """Return a normalized result without attempting external inference."""
        normalized = message.casefold().strip()
        if not normalized:
            return Sentiment.UNKNOWN
        if any(term in normalized for term in self._NEGATIVE_TERMS):
            return Sentiment.NEGATIVE
        if any(term in normalized for term in self._POSITIVE_TERMS):
            return Sentiment.POSITIVE
        return Sentiment.NEUTRAL


class OciSentimentAnalyzer:
    """Adapt an injected, reviewed OCI client while remaining SDK-independent.

    The injected callable may return a shared ``Sentiment``, a matching string, or an
    object/mapping with a ``sentiment`` value. Failures intentionally become UNKNOWN so
    callers can continue safely without exposing provider details.
    """

    def __init__(self, client: Callable[[str], Any]) -> None:
        self._client = client

    def analyze(self, message: str) -> Sentiment:
        """Call the injected client and normalize only recognized sentiment values."""
        try:
            result = self._client(message)
        except Exception:
            return Sentiment.UNKNOWN

        if isinstance(result, Sentiment):
            return result
        if isinstance(result, dict):
            result = result.get("sentiment")
        else:
            result = getattr(result, "sentiment", result)
        try:
            return Sentiment(str(result).strip().upper())
        except (TypeError, ValueError):
            return Sentiment.UNKNOWN
