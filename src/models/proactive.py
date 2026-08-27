"""Shared contracts for proactive-support output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    """Normalized customer sentiment values."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class ArticleReference(BaseModel):
    """A lightweight reference to a recommended knowledge article."""

    article_id: str = Field(min_length=1)
    title: Optional[str] = None
    source: Optional[str] = None


class ProactiveAnalysis(BaseModel):
    """Implementation-independent proactive signals for conversation consumers."""

    sentiment: Sentiment = Sentiment.UNKNOWN
    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    escalation_required: bool = False
    reason: Optional[str] = None
    recommended_articles: list[ArticleReference] = Field(default_factory=list)
