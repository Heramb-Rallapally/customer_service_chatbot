"""Deterministic structured-context extraction and clarification helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.models import ConversationState, ResolutionStatus, Severity

from .intent import ConversationIntent


@dataclass(frozen=True)
class Clarification:
    """The next retrieval-relevant field and targeted question."""

    field: str
    question: str


class ContextUpdater:
    """Extract a deliberately small set of reliable support attributes."""

    _named_vpn = re.compile(
        r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,2}\s+VPN)\b"
    )
    _explicit_product = re.compile(
        r"\b(?:using|use|client is|product is)\s+([\w.-]+(?:\s+[\w.-]+){0,3})",
        re.IGNORECASE,
    )
    _explicit_version = re.compile(
        r"\b(?:version|ver\.?|v)\s*([0-9]+(?:\.[0-9A-Za-z-]+)+)\b",
        re.IGNORECASE,
    )
    _bare_version = re.compile(r"^\s*([0-9]+(?:\.[0-9A-Za-z-]+)+)\s*[.!]?\s*$")
    _authentication = re.compile(
        r"\b(authentication|authenticate|login|sign[ -]?in|credentials?|token)\b",
        re.IGNORECASE,
    )
    _failure = re.compile(
        r"\b(fail(?:ed|ing|ure)?|denied|invalid|expired|error)\b", re.IGNORECASE
    )
    _connectivity = re.compile(
        r"\b(can(?:not|'t) connect|not connecting|connection (?:failed|drops?|lost)|"
        r"disconnect(?:ed|ing)?)\b",
        re.IGNORECASE,
    )
    _generic_issue = re.compile(
        r"\b(isn't working|is not working|doesn't work|does not work|problem|issue)\b",
        re.IGNORECASE,
    )

    def update(
        self,
        state: ConversationState,
        message: str,
        intent: ConversationIntent,
    ) -> ConversationState:
        updated = state.model_copy(deep=True)
        normalized = " ".join(message.strip().split())

        product = self._extract_product(normalized, updated, intent)
        if product:
            updated.product = product

        version = self._extract_version(normalized, updated, intent)
        if version:
            updated.version = version

        issue_type, summary = self._extract_issue(normalized)
        if issue_type:
            updated.issue_type = issue_type
            updated.issue_summary = summary
        elif updated.issue_summary is None and self._generic_issue.search(normalized):
            updated.issue_summary = normalized.rstrip(".!?")

        severity = self._extract_severity(normalized)
        if severity is not None:
            updated.severity = severity

        return updated

    def _extract_product(
        self,
        message: str,
        state: ConversationState,
        intent: ConversationIntent,
    ) -> Optional[str]:
        match = self._named_vpn.search(message)
        if match and match.group(1).lower() not in {"my vpn", "the vpn"}:
            candidate = match.group(1).strip()
            return re.sub(r"^(?:my|the)\s+", "", candidate, flags=re.IGNORECASE)

        match = self._explicit_product.search(message)
        if match:
            candidate = match.group(1).strip(" .,!?")
            if candidate:
                return candidate

        if (
            state.product is None
            and intent is ConversationIntent.CLARIFICATION_RESPONSE
            and len(message.split()) <= 5
            and not self._explicit_version.search(message)
            and not self._bare_version.match(message)
            and not self._failure.search(message)
        ):
            lowered = message.lower().strip(" .,!?")
            if lowered not in {"unknown", "not sure", "i don't know", "i do not know"}:
                return message.strip(" .,!?")
        return None

    def _extract_version(
        self,
        message: str,
        state: ConversationState,
        intent: ConversationIntent,
    ) -> Optional[str]:
        match = self._explicit_version.search(message)
        if match:
            return match.group(1)
        if state.product and state.version is None and intent is ConversationIntent.CLARIFICATION_RESPONSE:
            match = self._bare_version.match(message)
            if match:
                return match.group(1)
            if message.lower().strip(" .,!?") in {
                "unknown",
                "not sure",
                "i don't know",
                "i do not know",
            }:
                return "unknown"
        return None

    def _extract_issue(self, message: str) -> tuple[Optional[str], Optional[str]]:
        if self._authentication.search(message) and (
            self._failure.search(message) or self._generic_issue.search(message)
        ):
            return "authentication", "authentication failed"
        if self._connectivity.search(message):
            return "connectivity", message.rstrip(".!?")
        if re.search(r"\b(error\s*(?:code)?\s*[A-Z0-9-]+)\b", message, re.IGNORECASE):
            return "error", message.rstrip(".!?")
        return None, None

    @staticmethod
    def _extract_severity(message: str) -> Optional[Severity]:
        lowered = message.lower()
        if any(term in lowered for term in ("critical", "production down", "outage")):
            return Severity.CRITICAL
        if any(term in lowered for term in ("urgent", "high severity", "business blocked")):
            return Severity.HIGH
        if "low severity" in lowered:
            return Severity.LOW
        return None


class ClarificationPlanner:
    """Select one missing field that most improves retrieval quality."""

    _information_request = re.compile(
        r"^\s*(?:what|which|where|when|who|why|how|does|do|is|are|can|could|"
        r"would|should|tell\s+me|explain|describe|list)\b",
        re.IGNORECASE,
    )
    _factual_information_request = re.compile(
        r"^\s*(?:what\s+(?:is|are|does)|which|where|when|who|why|"
        r"tell\s+me\s+about|explain|describe|list)\b",
        re.IGNORECASE,
    )
    _word = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._+-]*")
    _non_subject_words = frozenset(
        {
            "a",
            "about",
            "an",
            "and",
            "anything",
            "are",
            "at",
            "be",
            "can",
            "could",
            "describe",
            "do",
            "does",
            "error",
            "explain",
            "fix",
            "for",
            "from",
            "give",
            "happen",
            "happening",
            "help",
            "how",
            "i",
            "information",
            "is",
            "issue",
            "it",
            "list",
            "me",
            "my",
            "next",
            "of",
            "on",
            "our",
            "please",
            "problem",
            "should",
            "something",
            "that",
            "the",
            "these",
            "this",
            "those",
            "to",
            "try",
            "us",
            "we",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "would",
            "wrong",
            "you",
            "your",
        }
    )

    def next_question(
        self,
        state: ConversationState,
        *,
        current_message: Optional[str] = None,
    ) -> Optional[Clarification]:
        if current_message and self.is_self_contained_knowledge_request(current_message):
            return None
        if not state.product:
            return Clarification(
                field="product",
                question="Which product or client are you using?",
            )
        if not state.version:
            return Clarification(
                field="version",
                question=f"What version of {state.product} are you using?",
            )
        if not state.issue_type:
            return Clarification(
                field="issue_type",
                question="What error message or specific behavior are you seeing?",
            )
        return None

    def is_self_contained_knowledge_request(self, message: str) -> bool:
        """Return whether a request names enough subject matter for knowledge search.

        Missing product/version fields are useful for troubleshooting, but they are
        not prerequisites for answering a specific informational question. Generic
        referential requests such as ``How do I fix this?`` contain no subject token
        and continue through the normal clarification sequence.
        """

        normalized = " ".join(message.strip().split())
        if not normalized:
            return False
        starts_as_question = self._information_request.search(normalized) is not None
        if not starts_as_question and not normalized.endswith("?"):
            return False
        subject_words = [
            word.casefold()
            for word in self._word.findall(normalized)
            if word.casefold() not in self._non_subject_words
        ]
        minimum_subject_words = 1 if starts_as_question else 2
        return len(subject_words) >= minimum_subject_words

    def is_informational_knowledge_request(self, message: str) -> bool:
        """Return whether a self-contained request asks for facts, not a fix."""

        return bool(
            self._factual_information_request.search(message)
            and self.is_self_contained_knowledge_request(message)
        )


class RetrievalQueryBuilder:
    """Build a focused query and metadata filters from structured state."""

    def build(self, state: ConversationState, current_message: str) -> tuple[str, dict[str, str]]:
        parts = [state.product or ""]
        if state.version and state.version.lower() != "unknown":
            parts.append(state.version)
        parts.append(state.issue_summary or state.issue_type or "")
        query = " ".join(part.strip() for part in parts if part and part.strip())
        if not query:
            query = " ".join(current_message.strip().split())

        filters: dict[str, str] = {}
        if state.product:
            filters["product"] = state.product
        if state.version and state.version.lower() != "unknown":
            filters["version"] = state.version
        if state.issue_type:
            filters["issue_type"] = state.issue_type
        if state.severity:
            filters["severity"] = state.severity.value
        return query, filters


def reset_for_new_request(state: ConversationState) -> ConversationState:
    """Keep history and identity while clearing issue-specific state."""

    return state.model_copy(
        deep=True,
        update={
            "product": None,
            "version": None,
            "issue_type": None,
            "issue_summary": None,
            "severity": None,
            "resolution_status": ResolutionStatus.NEW,
            "troubleshooting_steps": [],
            "attempted_steps": [],
        },
    )
