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
