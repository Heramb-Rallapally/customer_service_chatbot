"""Make conservative, explainable escalation decisions without ticket routing.

The function emits one stable primary reason that fits the shared proactive contract.
Future workflow code may route an already-approved escalation, but must not add routing
or persistence side effects to this decision-only module.
"""

from __future__ import annotations

from src.models.conversation import ConversationState, ResolutionStatus
from src.models.knowledge import Severity

from .scoring import has_human_support_request, has_repeated_failure_language


def escalation_reason(
    message: str,
    frustration_score: float,
    conversation: ConversationState | None = None,
    *,
    unsupported_issue: bool = False,
) -> str | None:
    """Return the strongest applicable stable escalation reason, if any."""
    if conversation is not None and conversation.resolution_status is ResolutionStatus.ESCALATED:
        return "already_escalated"
    if has_human_support_request(message):
        return "human_support_requested"
    if unsupported_issue:
        return "unsupported_issue"
    if conversation is not None and conversation.severity in {Severity.CRITICAL, Severity.HIGH}:
        return "high_severity"
    if frustration_score >= 0.70:
        return "high_frustration"
    if (
        has_repeated_failure_language(message)
        and conversation is not None
        and len(conversation.attempted_steps) >= 1
    ):
        return "repeated_failed_troubleshooting"
    return None
