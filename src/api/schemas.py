"""HTTP request models owned by the API layer."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Validated input accepted by the chat endpoint."""

    conversation_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    user_id: Optional[str] = None

    @field_validator("conversation_id", "user_message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        """Keep identifiers and customer messages meaningful."""

        if not value.strip():
            raise ValueError("must not be blank")
        return value


class FeedbackRequest(BaseModel):
    """Authenticated customer feedback for a conversation outcome."""

    conversation_id: str = Field(min_length=1)
    rating: str = Field(pattern="^(positive|negative)$")
    comment: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("conversation_id")
    @classmethod
    def reject_blank_conversation_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class FeedbackResponse(BaseModel):
    """Deliberately small acknowledgement without internal storage details."""

    accepted: bool = True
