"""Metadata filters shared by retrievers and vector-store adapters."""

from __future__ import annotations

from collections.abc import Mapping
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

    @classmethod
    def from_conversation_mapping(cls, filters: Mapping[str, str]) -> "RetrievalFilters":
        """Convert the conversation port's exact filter mapping safely.

        The conversation engine owns only product, version, issue type, and
        severity. Rejecting unknown keys avoids silently dropping a constraint
        that could otherwise return unrelated support guidance.
        """

        supported = {"product", "version", "issue_type", "severity"}
        unsupported = set(filters) - supported
        if unsupported:
            raise ValueError(
                "unsupported conversation retrieval filters: "
                + ", ".join(sorted(unsupported))
            )
        if any(not isinstance(value, str) for value in filters.values()):
            raise ValueError("conversation retrieval filter values must be strings")
        values = {name: value.strip() for name, value in filters.items() if value.strip()}
        if "severity" in values:
            values["severity"] = values["severity"].upper()
        return cls(**values)
