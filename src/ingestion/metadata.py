"""Small, transparent heuristics for normalizing support knowledge metadata."""

from __future__ import annotations

import re
from typing import Any

from src.models import Severity

from .models import IngestionRecord

_VERSION = re.compile(r"\b(?:version|ver\.?|v)\s*(\d+(?:\.\d+){0,3})\b", re.IGNORECASE)
_ISSUE_PATTERNS = {
    "connectivity": ("connect", "connection", "network", "vpn", "timeout"),
    "authentication": ("login", "sign in", "password", "authentication", "credential"),
    "installation": ("install", "setup", "upgrade", "update", "deploy"),
    "performance": ("slow", "latency", "performance", "freeze", "crash"),
    "configuration": ("config", "setting", "configure", "permission"),
}
_SEVERITY_PATTERNS = {
    Severity.CRITICAL: ("critical", "outage", "data loss", "security incident"),
    Severity.HIGH: ("high severity", "urgent", "production down", "service unavailable"),
    Severity.MEDIUM: ("medium severity", "degraded", "intermittent"),
    Severity.LOW: ("low severity", "how to", "minor"),
}
_RESOLUTION_PATTERNS = {
    "workaround": ("workaround", "temporary fix"),
    "configuration_change": ("configure", "configuration", "setting"),
    "upgrade": ("upgrade", "update to", "patch"),
    "restart": ("restart", "reboot", "reload"),
}


class MetadataExtractor:
    """Extract optional normalized metadata while preserving supplied values."""

    def __init__(self, known_products: tuple[str, ...] = ()) -> None:
        self._known_products = tuple(product for product in known_products if product.strip())

    def extract(self, record: IngestionRecord, content: str) -> dict[str, Any]:
        """Return metadata fields suitable for ``KnowledgeDocument`` construction."""

        supplied = record.metadata
        product = self._first_text(supplied, "product") or self._detect_product(content)
        issue_type = self._first_text(supplied, "issue_type") or self._detect_pattern(content, _ISSUE_PATTERNS)
        severity = self._parse_severity(supplied.get("severity")) or self._detect_severity(content)
        resolution_category = self._first_text(supplied, "resolution_category") or self._detect_pattern(
            content, _RESOLUTION_PATTERNS
        )
        version = self._first_text(supplied, "version") or self._detect_version(content)
        return {
            "product": product,
            "issue_type": issue_type,
            "severity": severity,
            "resolution_category": resolution_category,
            "version": version,
        }

    @staticmethod
    def _first_text(metadata: dict[str, Any], key: str) -> str | None:
        value = metadata.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _detect_product(self, content: str) -> str | None:
        lower_content = content.lower()
        return next((product for product in self._known_products if product.lower() in lower_content), None)

    @staticmethod
    def _detect_pattern(content: str, patterns: dict[str, tuple[str, ...]]) -> str | None:
        lower_content = content.lower()
        return next((label for label, terms in patterns.items() if any(term in lower_content for term in terms)), None)

    @staticmethod
    def _parse_severity(value: Any) -> Severity | None:
        if isinstance(value, Severity):
            return value
        if isinstance(value, str):
            try:
                return Severity(value.upper())
            except ValueError:
                return None
        return None

    def _detect_severity(self, content: str) -> Severity | None:
        lower_content = content.lower()
        return next(
            (severity for severity, terms in _SEVERITY_PATTERNS.items() if any(term in lower_content for term in terms)),
            None,
        )

    @staticmethod
    def _detect_version(content: str) -> str | None:
        match = _VERSION.search(content)
        return match.group(1) if match else None
