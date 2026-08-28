"""Dependency boundaries used by the conversation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

from src.models import (
    ConversationMessage,
    ConversationState,
    ProactiveAnalysis,
    RetrievalResult,
)


class Retriever(Protocol):
    """Search boundary implemented by the retrieval team."""

    def search(
        self,
        *,
        query: str,
        filters: Mapping[str, str],
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        """Return relevant results ordered from highest to lowest confidence."""


@dataclass(frozen=True)
class GenerationContext:
    """Bounded, structured input supplied to an LLM implementation."""

    system_instructions: str
    conversation: ConversationState
    recent_messages: tuple[ConversationMessage, ...]
    conversation_summary: Optional[str]
    retrieved_knowledge: tuple[RetrievalResult, ...]
    proactive_analysis: ProactiveAnalysis
    current_user_message: str
    excluded_steps: tuple[str, ...]
    cautious: bool = False


@dataclass(frozen=True)
class GeneratedResponse:
    """Content returned by an LLM implementation before orchestration metadata."""

    message: str
    suggested_actions: tuple[str, ...] = ()
    confidence: Optional[float] = None


class LLMService(Protocol):
    """Grounded response-generation boundary implemented by an OCI adapter."""

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        """Generate a response using only the supplied knowledge and context."""


class ProactiveService(Protocol):
    """Proactive-analysis boundary implemented by the proactive team."""

    def analyze(
        self, *, message: str, conversation: ConversationState
    ) -> ProactiveAnalysis:
        """Return structured proactive signals without controlling the workflow."""


class ConversationMemory(Protocol):
    """Persistence boundary for conversation state and a future summary."""

    def load(self, conversation_id: str) -> Optional[ConversationState]:
        """Load a conversation or return ``None`` when it does not exist."""

    def save(self, state: ConversationState) -> None:
        """Persist the latest conversation state."""

    def get_summary(self, conversation_id: str) -> Optional[str]:
        """Return an optional bounded summary of older conversation context."""


@dataclass(frozen=True)
class ConversationSnapshot:
    """A loaded state paired with its persistence version."""

    state: ConversationState
    version: int


class VersionedConversationMemory(ConversationMemory, Protocol):
    """Optional optimistic-concurrency capability for durable memory adapters."""

    def load_with_version(
        self, conversation_id: str
    ) -> Optional[ConversationSnapshot]:
        """Load state and the version required for a conflict-safe save."""

    def save_with_version(self, state: ConversationState, *, expected_version: int) -> int:
        """Save state only when its stored version matches ``expected_version``."""
