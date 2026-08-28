"""Typed conversation-boundary errors for API adapters."""

from __future__ import annotations


class ConversationOwnershipError(ValueError):
    """Raised when a caller does not own a user-bound conversation.

    Subclassing ``ValueError`` preserves the existing engine-level validation
    behavior for direct callers while allowing HTTP adapters to map this case
    without inspecting an exception message.
    """

