"""Conversation memory implementations."""

from __future__ import annotations

from threading import RLock
from typing import Optional

from src.models import ConversationState

from .interfaces import ConversationSnapshot


class InMemoryConversationMemory:
    """Thread-safe, process-local memory for tests and local development only."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}
        self._versions: dict[str, int] = {}
        self._summaries: dict[str, str] = {}
        self._lock = RLock()

    def load(self, conversation_id: str) -> Optional[ConversationState]:
        with self._lock:
            state = self._states.get(conversation_id)
            return state.model_copy(deep=True) if state is not None else None

    def save(self, state: ConversationState) -> None:
        with self._lock:
            self._states[state.conversation_id] = state.model_copy(deep=True)
            self._versions[state.conversation_id] = self._versions.get(state.conversation_id, 0) + 1

    def load_with_version(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                return None
            return ConversationSnapshot(
                state=state.model_copy(deep=True),
                version=self._versions[conversation_id],
            )

    def save_with_version(self, state: ConversationState, *, expected_version: int) -> int:
        """Provide the same optimistic-concurrency behavior for local tests."""

        from .memory_exceptions import ConversationConflictError

        with self._lock:
            current_version = self._versions.get(state.conversation_id, 0)
            if current_version != expected_version:
                raise ConversationConflictError("Conversation was updated by another request")
            next_version = current_version + 1
            self._states[state.conversation_id] = state.model_copy(deep=True)
            self._versions[state.conversation_id] = next_version
            return next_version

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
