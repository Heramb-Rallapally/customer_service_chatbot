"""Expose proactive-support orchestration and safe provider boundaries.

This package gives conversation consumers one side-effect-free service and narrow
injection protocols. Keep shared Pydantic contracts in ``src.models.proactive`` so
cross-module representations remain canonical.
"""

from .interfaces import (
    HistoryProvider,
    RecommendationProvider,
    SentimentAnalyzer,
    UnsupportedIssueDetector,
)
from .service import ProactiveSupportService
from .sentiment import OciSentimentAnalyzer, RuleBasedSentimentAnalyzer
from .providers import (
    ConversationMemoryHistoryProvider,
    RetrievalEvidenceProvider,
    SupportLevel,
)

__all__ = [
    "HistoryProvider",
    "ConversationMemoryHistoryProvider",
    "OciSentimentAnalyzer",
    "ProactiveSupportService",
    "RecommendationProvider",
    "RetrievalEvidenceProvider",
    "RuleBasedSentimentAnalyzer",
    "SentimentAnalyzer",
    "UnsupportedIssueDetector",
    "SupportLevel",
]
