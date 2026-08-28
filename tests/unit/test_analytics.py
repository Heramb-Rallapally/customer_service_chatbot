"""Credential-free tests for support analytics and feedback collection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.analytics import (
    FeedbackRating,
    InMemoryAnalyticsEventSink,
    SupportEvent,
    SupportEventType,
    to_evaluation_records,
)
from src.api import ChatApplicationService, ChatRequest
from src.models import ChatResponse, Citation, ConversationState, ResolutionStatus
from src.ui import summarize_events


class StatefulConversation:
    def __init__(self) -> None:
        self.state = ConversationState(
            conversation_id="conversation-1",
            user_id="user-1",
            issue_type="connectivity",
            resolution_status=ResolutionStatus.RESOLVED,
        )

    def handle_message(self, **_kwargs: object) -> ChatResponse:
        return ChatResponse(
            message="Grounded response",
            citations=[Citation(source="Official guide", document_id="guide-1")],
            suggested_actions=["Check the documented setting"],
            confidence=0.8,
        )

    def get_state(self, _conversation_id: str) -> ConversationState:
        return self.state.model_copy(deep=True)


def test_support_event_validates_bounds_and_optional_fields() -> None:
    event = SupportEvent(conversation_id="conversation-1")
    assert event.user_id is None
    assert event.response_confidence is None
    with pytest.raises(ValidationError):
        SupportEvent(conversation_id="conversation-1", response_confidence=1.1)
    with pytest.raises(ValidationError):
        SupportEvent(conversation_id="conversation-1", feedback_rating="invalid")
    with pytest.raises(ValidationError):
        SupportEvent(conversation_id="conversation-1", timestamp=datetime(2026, 1, 1))


def test_in_memory_sink_preserves_order_returns_copies_and_can_clear() -> None:
    sink = InMemoryAnalyticsEventSink()
    first = SupportEvent(conversation_id="one")
    second = SupportEvent(conversation_id="two")
    sink.record(first)
    sink.record(second)

    events = sink.events()
    assert [event.conversation_id for event in events] == ["one", "two"]
    assert events[0] is not first
    sink.clear()
    assert sink.events() == ()


def test_in_memory_sink_accepts_concurrent_recording() -> None:
    sink = InMemoryAnalyticsEventSink()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda value: sink.record(SupportEvent(conversation_id=str(value))), range(20)))
    assert len(sink.events()) == 20


def test_chat_outcome_and_feedback_capture_known_outcome_data() -> None:
    sink = InMemoryAnalyticsEventSink()
    service = ChatApplicationService(StatefulConversation(), analytics_sink=sink)
    response = service.chat(
        ChatRequest(conversation_id="conversation-1", user_id="user-1", user_message="Help")
    )
    service.record_feedback(
        conversation_id="conversation-1",
        user_id="user-1",
        rating=FeedbackRating.POSITIVE,
        comment="Useful guidance",
    )

    assert response.message == "Grounded response"
    chat_event, feedback_event = sink.events()
    assert chat_event.event_type is SupportEventType.CHAT_OUTCOME
    assert chat_event.resolution_status is ResolutionStatus.RESOLVED
    assert chat_event.response_confidence == 0.8
    assert chat_event.response_time_ms is not None
    assert chat_event.suggested_actions_count == 1
    assert chat_event.citation_count == 1
    assert feedback_event.feedback_rating is FeedbackRating.POSITIVE
    assert feedback_event.feedback_comment == "Useful guidance"


def test_analytics_failures_do_not_break_chat_or_feedback() -> None:
    class FailingSink:
        def record(self, _event: SupportEvent) -> None:
            raise RuntimeError("analytics database details")

    service = ChatApplicationService(StatefulConversation(), analytics_sink=FailingSink())
    assert service.chat(ChatRequest(conversation_id="conversation-1", user_id="user-1", user_message="Help")).message
    service.record_feedback(
        conversation_id="conversation-1", user_id="user-1", rating=FeedbackRating.NEGATIVE
    )


def test_feedback_requires_the_conversation_owner() -> None:
    service = ChatApplicationService(StatefulConversation(), analytics_sink=InMemoryAnalyticsEventSink())
    with pytest.raises(ValueError):
        service.record_feedback(
            conversation_id="conversation-1", user_id="other-user", rating=FeedbackRating.NEGATIVE
        )


def test_aggregation_uses_correct_denominators_and_ignores_missing_values() -> None:
    summary = summarize_events(
        [
            SupportEvent(
                conversation_id="one",
                resolution_status=ResolutionStatus.RESOLVED,
                escalation_required=True,
                response_time_ms=100,
                response_confidence=0.9,
                issue_type="connectivity",
            ),
            SupportEvent(
                conversation_id="two",
                resolution_status=ResolutionStatus.ESCALATED,
                escalation_required=False,
                feedback_rating=FeedbackRating.POSITIVE,
            ),
            SupportEvent(
                conversation_id="two",
                event_type=SupportEventType.FEEDBACK,
                feedback_rating=FeedbackRating.NEGATIVE,
            ),
        ]
    )
    assert summary.conversation_count == 2
    assert summary.event_count == 3
    assert summary.resolution_rate == 0.5
    assert summary.escalation_rate == 0.5
    assert summary.average_response_time_ms == 100
    assert summary.average_confidence == 0.9
    assert summary.feedback_count == 2
    assert summary.positive_feedback_rate == 0.5
    assert summary.negative_feedback_rate == 0.5
    assert summary.common_issues == {"connectivity": 1}


def test_evaluation_records_join_feedback_without_conversation_text() -> None:
    records = to_evaluation_records(
        [
            SupportEvent(
                conversation_id="conversation-1",
                issue_type="connectivity",
                resolution_status=ResolutionStatus.RESOLVED,
                response_confidence=0.8,
            ),
            SupportEvent(
                conversation_id="conversation-1",
                event_type=SupportEventType.FEEDBACK,
                feedback_rating=FeedbackRating.POSITIVE,
                feedback_comment="not included in records",
            ),
        ]
    )
    assert records[0].feedback_rating is FeedbackRating.POSITIVE
    assert not hasattr(records[0], "feedback_comment")
