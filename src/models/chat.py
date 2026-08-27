"""Shared response contracts for conversation, API, and UI consumers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .proactive import ArticleReference


class Citation(BaseModel):
    """A source reference supporting a chat response."""

    source: str = Field(min_length=1)
    document_id: Optional[str] = None
    excerpt: Optional[str] = None


class ChatResponse(BaseModel):
    """Grounded response contract shared by conversation, API, and UI layers."""

    message: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    related_articles: list[ArticleReference] = Field(default_factory=list)
