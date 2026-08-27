"""Shared contracts for retrieval output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A scored knowledge item returned by a retrieval implementation."""

    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

