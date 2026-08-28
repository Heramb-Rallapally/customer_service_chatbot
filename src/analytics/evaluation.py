"""Dependency-free conversion of observed events into offline evaluation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .events import FeedbackRating, SupportEvent, SupportEventType


@dataclass(frozen=True)
class SupportEvaluationRecord:
    """Outcome-only record for future offline evaluation in Step 8."""

    conversation_id: str
    issue_type: Optional[str]
    resolution_status: Optional[str]
    escalation_required: bool
    response_confidence: Optional[float]
    response_time_ms: Optional[float]
    feedback_rating: Optional[FeedbackRating]


def to_evaluation_records(events: Iterable[SupportEvent]) -> Sequence[SupportEvaluationRecord]:
    """Join feedback to its conversation's latest known chat outcome.

    This intentionally emits no raw conversation text and performs no model
    update. It is a stable input for a later offline evaluation workflow.
    """

    outcomes: dict[str, SupportEvent] = {}
    feedback: dict[str, FeedbackRating] = {}
    for event in events:
        if event.event_type is SupportEventType.CHAT_OUTCOME:
            outcomes[event.conversation_id] = event
        elif event.event_type is SupportEventType.FEEDBACK and event.feedback_rating:
            feedback[event.conversation_id] = event.feedback_rating
    return tuple(
        SupportEvaluationRecord(
            conversation_id=event.conversation_id,
            issue_type=event.issue_type,
            resolution_status=(
                event.resolution_status.value if event.resolution_status else None
            ),
            escalation_required=event.escalation_required,
            response_confidence=event.response_confidence,
            response_time_ms=event.response_time_ms,
            feedback_rating=feedback.get(event.conversation_id),
        )
        for event in outcomes.values()
    )
