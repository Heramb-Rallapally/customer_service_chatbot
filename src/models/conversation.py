"""Shared contracts for multi-turn conversation state."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .knowledge import Severity


class MessageRole(str, Enum):
    """Role of a participant in a stored conversation message."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class ConversationMessage(BaseModel):
    """A single structured message retained in conversation state."""

    role: MessageRole
    content: str = Field(min_length=1)


class ResolutionStatus(str, Enum):
    """Lifecycle states shared by conversation consumers."""

    NEW = "NEW"
    UNDERSTANDING = "UNDERSTANDING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    READY_TO_RESOLVE = "READY_TO_RESOLVE"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ESCALATED = "ESCALATED"


class ConversationState(BaseModel):
    """State exchanged across turns without conversation-engine behavior."""

    conversation_id: str = Field(min_length=1)
    user_id: Optional[str] = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    product: Optional[str] = None
    version: Optional[str] = None
    issue_type: Optional[str] = None
    issue_summary: Optional[str] = None
    severity: Optional[Severity] = None
    resolution_status: ResolutionStatus = ResolutionStatus.NEW
    troubleshooting_steps: list[str] = Field(default_factory=list)
    attempted_steps: list[str] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0)
