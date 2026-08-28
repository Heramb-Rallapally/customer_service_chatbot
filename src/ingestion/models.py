"""Internal, dependency-light models used before knowledge reaches retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Knowledge source categories supported by the ingestion pipeline."""

    PRODUCT_DOCUMENTATION = "product_documentation"
    FAQ = "faq"
    HISTORICAL_TICKET = "historical_ticket"
    KNOWLEDGE_BASE = "knowledge_base"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    VIDEO_TRANSCRIPT = "video_transcript"


class IngestionRecord(BaseModel):
    """A raw knowledge item with optional source-provided metadata."""

    source: str = Field(min_length=1)
    content: str
    source_type: SourceType = SourceType.PRODUCT_DOCUMENTATION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def strip_source(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source must not be blank")
        return value


class ChunkingConfig(BaseModel):
    """Configuration for deterministic character-based chunking."""

    chunk_size: int = Field(default=1_000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    def model_post_init(self, __context: Any) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
