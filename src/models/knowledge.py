"""Shared contracts for knowledge items."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Normalized support issue severity."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class KnowledgeDocument(BaseModel):
    """A knowledge item ready to cross ingestion and retrieval boundaries."""

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    product: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[Severity] = None
    resolution_category: Optional[str] = None
    version: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
