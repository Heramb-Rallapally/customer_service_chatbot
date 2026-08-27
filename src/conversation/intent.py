"""Small deterministic intent representation for conversation orchestration."""

from __future__ import annotations

import re
from enum import Enum

from src.models import ConversationState, ResolutionStatus


class ConversationIntent(str, Enum):
    """Intent categories needed by the conversation workflow."""

    NEW_ISSUE = "NEW_ISSUE"
    FOLLOW_UP = "FOLLOW_UP"
    CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"
    TROUBLESHOOTING_RESULT = "TROUBLESHOOTING_RESULT"
    RESOLUTION_CONFIRMATION = "RESOLUTION_CONFIRMATION"
    NEW_REQUEST = "NEW_REQUEST"
    ESCALATION_REQUEST = "ESCALATION_REQUEST"


class IntentDetector:
    """Classify workflow intent using explicit, auditable rules."""

    _escalation = re.compile(
        r"\b(human|live agent|support agent|representative|escalat(?:e|ion)|supervisor)\b",
        re.IGNORECASE,
    )
    _new_request = re.compile(
        r"\b(new|another|different)\s+(issue|problem|request)\b", re.IGNORECASE
    )
    _failed_step = re.compile(
        r"\b(already tried|tried that|did not work|didn't work|does not work|"
        r"doesn't work|still (?:doesn't|does not|isn't|is not) work(?:ing)?|"
        r"not working)\b",
        re.IGNORECASE,
    )
    _resolved = re.compile(
        r"\b(yes[,! ]+(?:it|that) works?|works? now|working now|that worked|"
        r"resolved|fixed)\b",
        re.IGNORECASE,
    )

    def detect(self, message: str, state: ConversationState) -> ConversationIntent:
        normalized = message.strip()
        if self._escalation.search(normalized):
            return ConversationIntent.ESCALATION_REQUEST
        if self._new_request.search(normalized) and state.resolution_status in {
            ResolutionStatus.RESOLVED,
            ResolutionStatus.ESCALATED,
        }:
            return ConversationIntent.NEW_REQUEST
        if self._failed_step.search(normalized):
            return ConversationIntent.TROUBLESHOOTING_RESULT
        if (
            state.troubleshooting_steps
            and self._resolved.search(normalized)
            and state.resolution_status
            in {
                ResolutionStatus.AWAITING_CONFIRMATION,
                ResolutionStatus.TROUBLESHOOTING,
                ResolutionStatus.UNRESOLVED,
                ResolutionStatus.ESCALATED,
            }
        ):
            return ConversationIntent.RESOLUTION_CONFIRMATION
        if not state.messages and state.resolution_status is ResolutionStatus.NEW:
            return ConversationIntent.NEW_ISSUE
        if state.resolution_status is ResolutionStatus.NEEDS_CLARIFICATION:
            return ConversationIntent.CLARIFICATION_RESPONSE
        return ConversationIntent.FOLLOW_UP

