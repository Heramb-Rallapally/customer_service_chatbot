"""Typed, privacy-conscious operational events for support analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.models import ResolutionStatus


class SupportEventType(str, Enum):
    """Kinds of observable support outcomes collected in Step 7."""

    CHAT_OUTCOME = "chat_outcome"
    FEEDBACK = "feedback"


class FeedbackRating(str, Enum):
    """Supported helpfulness ratings."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class SupportEvent(BaseModel):
    """A structured event without raw customer or assistant conversation text."""

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: SupportEventType = SupportEventType.CHAT_OUTCOME
    conversation_id: str = Field(min_length=1)
    user_id: Optional[str] = None
    issue_type: Optional[str] = None
    resolution_status: Optional[ResolutionStatus] = None
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    response_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    response_time_ms: Optional[float] = Field(default=None, ge=0.0)
    retrieval_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    feedback_rating: Optional[FeedbackRating] = None
    feedback_comment: Optional[str] = Field(default=None, max_length=2000)
    feedback_type: Optional[str] = Field(default=None, max_length=100)
    suggested_actions_count: int = Field(default=0, ge=0)
    citation_count: int = Field(default=0, ge=0)

    @field_validator("conversation_id", "user_id", "issue_type", "escalation_reason", "feedback_type")
    @classmethod
    def reject_blank_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank when supplied")
        return normalized

    @field_validator("feedback_comment")
    @classmethod
    def normalize_feedback_comment(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("timestamp")
    @classmethod
    def require_unambiguous_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)
