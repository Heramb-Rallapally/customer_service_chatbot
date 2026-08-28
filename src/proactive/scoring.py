"""Calculate deterministic, bounded frustration signals for proactive support.

This module combines message and existing conversation-state evidence without mutating
that state. Thresholds are intentionally explicit and should be tuned only against a
reviewed evaluation set, not incidental production anecdotes.
"""

from __future__ import annotations

from src.models.conversation import ConversationState, ResolutionStatus
from src.models.knowledge import Severity
from src.models.proactive import Sentiment

_NEGATIVE_TERMS = ("angry", "frustrated", "terrible", "useless", "disappointed")
_FAILURE_TERMS = (
    "again",
    "already tried",
    "didn't work",
    "does not work",
    "not working",
    "still failing",
    "still not",
)
_HUMAN_SUPPORT_TERMS = ("human", "live agent", "representative", "real person")


def has_human_support_request(message: str) -> bool:
    """Return whether the customer explicitly asks to speak with human support."""
    normalized = message.casefold()
    return any(term in normalized for term in _HUMAN_SUPPORT_TERMS)


def has_repeated_failure_language(message: str) -> bool:
    """Return whether this message reports failed or repeated troubleshooting."""
    normalized = message.casefold()
    return any(term in normalized for term in _FAILURE_TERMS)


def calculate_frustration_score(
    message: str,
    sentiment: Sentiment,
    conversation: ConversationState | None = None,
) -> float:
    """Return a deterministic score in the inclusive range 0.0 through 1.0."""
    normalized = message.casefold()
    score = 0.0

    if sentiment is Sentiment.NEGATIVE:
        score += 0.30
    if any(term in normalized for term in _NEGATIVE_TERMS):
        score += 0.15
    if has_repeated_failure_language(message):
        score += 0.25
    if has_human_support_request(message):
        score += 0.30

    if conversation is not None:
        if conversation.severity is Severity.CRITICAL:
            score += 0.20
        elif conversation.severity is Severity.HIGH:
            score += 0.12
        attempted_count = len(conversation.attempted_steps)
        if attempted_count >= 2:
            score += 0.20
        elif attempted_count == 1:
            score += 0.08
        if conversation.turn_count >= 5:
            score += 0.12
        elif conversation.turn_count >= 3:
            score += 0.05
        if conversation.resolution_status is ResolutionStatus.UNRESOLVED:
            score += 0.20
        elif conversation.resolution_status is ResolutionStatus.ESCALATED:
            score += 0.30

    return max(0.0, min(1.0, score))
