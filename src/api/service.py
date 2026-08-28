"""Thin application service between HTTP routes and the conversation engine."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Optional, Protocol, Sequence

from src.analytics import (
    AnalyticsEventSink,
    FeedbackRating,
    NoOpAnalyticsEventSink,
    SupportEvent,
    SupportEventType,
)
from src.conversation import ConversationOwnershipError
from src.models import ChatResponse, ResolutionStatus

from .schemas import ChatRequest

logger = logging.getLogger(__name__)


class ConversationService(Protocol):
    """The API-facing contract supplied by the conversation module."""

    def handle_message(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """Process one user turn and return the shared response contract."""


class ConversationServiceUnavailableError(RuntimeError):
    """Expected failure while calling the conversation dependency."""


class ChatApplicationService:
    """Delegates chat requests without duplicating conversation behavior."""

    def __init__(
        self,
        conversation_service: ConversationService,
        *,
        analytics_sink: Optional[AnalyticsEventSink] = None,
    ) -> None:
        self._conversation_service = conversation_service
        self._analytics_sink = analytics_sink or NoOpAnalyticsEventSink()

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Forward the validated turn to the conversation engine."""

        started_at = perf_counter()
        response = self._conversation_service.handle_message(
            conversation_id=request.conversation_id,
            user_message=request.user_message,
            user_id=request.user_id,
        )
        response_time_ms = (perf_counter() - started_at) * 1000
        state = self._load_state(request.conversation_id)
        self._record_fields_safely(
            event_type=SupportEventType.CHAT_OUTCOME,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            issue_type=getattr(state, "issue_type", None),
            resolution_status=getattr(state, "resolution_status", None),
            escalation_required=response.escalation_required,
            response_confidence=response.confidence,
            response_time_ms=response_time_ms,
            suggested_actions_count=len(response.suggested_actions),
            citation_count=len(response.citations),
        )
        return response

    def resolution_status(self, conversation_id: str) -> Optional[str]:
        """Return state created by the just-authorized chat turn, when available."""

        state = self._load_state(conversation_id)
        status = getattr(state, "resolution_status", None)
        value = getattr(status, "value", status)
        return value if isinstance(value, str) and value else None

    def record_feedback(
        self,
        *,
        conversation_id: str,
        user_id: str,
        rating: FeedbackRating,
        comment: Optional[str] = None,
    ) -> None:
        """Record feedback only after verifying conversation ownership."""

        state = self._load_state(conversation_id)
        if state is None or getattr(state, "user_id", None) != user_id:
            raise ConversationOwnershipError("feedback conversation is not owned by user")
        self._record_fields_safely(
            event_type=SupportEventType.FEEDBACK,
            conversation_id=conversation_id,
            user_id=user_id,
            issue_type=getattr(state, "issue_type", None),
            resolution_status=getattr(state, "resolution_status", None),
            escalation_required=(
                getattr(state, "resolution_status", None)
                is ResolutionStatus.ESCALATED
            ),
            feedback_rating=rating,
            feedback_comment=comment,
            feedback_type="helpfulness",
        )

    def events_for_user(self, user_id: str) -> Sequence[SupportEvent]:
        """Return only the requesting user's event snapshots for the local UI."""

        reader = self._analytics_sink
        events = getattr(reader, "events", None)
        if not callable(events):
            return ()
        try:
            return tuple(
                event.model_copy(deep=True)
                for event in events()
                if event.user_id == user_id
            )
        except Exception:
            logger.warning("Analytics event read failed")
            return ()

    def _load_state(self, conversation_id: str) -> object:
        get_state = getattr(self._conversation_service, "get_state", None)
        if not callable(get_state):
            return None
        try:
            return get_state(conversation_id)
        except Exception:
            return None

    def _record_safely(self, event: SupportEvent) -> None:
        try:
            self._analytics_sink.record(event)
        except Exception:
            # Analytics is observability only. Do not log raw event content.
            logger.warning("Support analytics event could not be recorded")

    def _record_fields_safely(self, **fields: Any) -> None:
        """Build and write an event without making analytics a chat dependency."""

        try:
            self._record_safely(SupportEvent(**fields))
        except Exception:
            logger.warning("Support analytics event could not be prepared")
