"""Thin application service between HTTP routes and the conversation engine."""

from __future__ import annotations

from typing import Optional, Protocol

from src.models import ChatResponse

from .schemas import ChatRequest


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

    def __init__(self, conversation_service: ConversationService) -> None:
        self._conversation_service = conversation_service

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Forward the validated turn to the conversation engine."""

        return self._conversation_service.handle_message(
            conversation_id=request.conversation_id,
            user_message=request.user_message,
            user_id=request.user_id,
        )
