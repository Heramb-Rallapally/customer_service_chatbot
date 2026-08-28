"""Safe persistence errors exposed by conversation-memory adapters."""

from __future__ import annotations


class ConversationMemoryError(RuntimeError):
    """Base class for durable conversation-memory failures."""


class ConversationNotFoundError(ConversationMemoryError):
    """Raised when an operation requires a conversation that does not exist."""


class ConversationConflictError(ConversationMemoryError):
    """Raised when a stale writer would overwrite a newer conversation turn."""


class ConversationPersistenceError(ConversationMemoryError):
    """Raised when conversation state cannot be safely read or persisted."""
