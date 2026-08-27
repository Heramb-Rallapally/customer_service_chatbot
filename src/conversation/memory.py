"""Conversation memory implementations."""

from __future__ import annotations

from threading import RLock
from typing import Optional

from src.models import ConversationState


class InMemoryConversationMemory:
    """Thread-safe process-local memory suitable for tests and local integration."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._summaries: dict[str, str] = {}
        self._lock = RLock()

    def load(self, conversation_id: str) -> Optional[ConversationState]:
        with self._lock:
            state = self._states.get(conversation_id)
            return state.model_copy(deep=True) if state is not None else None

    def save(self, state: ConversationState) -> None:
        with self._lock:
            self._states[state.conversation_id] = state.model_copy(deep=True)

    def get_summary(self, conversation_id: str) -> Optional[str]:
        with self._lock:
            return self._summaries.get(conversation_id)

    def set_summary(self, conversation_id: str, summary: Optional[str]) -> None:
        """Store a summary without coupling the engine to a summarizer."""

        with self._lock:
            if summary:
                self._summaries[conversation_id] = summary
            else:
                self._summaries.pop(conversation_id, None)

