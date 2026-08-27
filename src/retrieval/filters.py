"""Metadata filters shared by retrievers and vector-store adapters."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from src.models import Severity


class RetrievalFilters(BaseModel):
    """Optional exact-match constraints applied to retrieved knowledge."""

    product: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[Severity] = None
    resolution_category: Optional[str] = None
    version: Optional[str] = None
    source: Optional[str] = None

    def as_metadata(self) -> dict[str, Any]:
        """Return only requested filters in vector-store-friendly form."""

        return {
            name: value.value if isinstance(value, Severity) else value
            for name, value in self.model_dump().items()
            if value is not None
        }
