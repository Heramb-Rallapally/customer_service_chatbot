"""Troubleshooting-step tracking without changing shared state contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.models import ConversationState, ResolutionStatus


def normalize_step(step: str) -> str:
    """Normalize a step for conservative duplicate detection."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", step.lower()).split())


class TroubleshootingTracker:
    """Track suggested and attempted steps in existing list fields."""

    def record_failed_attempt(
        self, state: ConversationState, user_message: str
    ) -> ConversationState:
        updated = state.model_copy(deep=True)
        attempted = self._find_referenced_step(updated, user_message)
        if attempted and normalize_step(attempted) not in {
            normalize_step(step) for step in updated.attempted_steps
        }:
            updated.attempted_steps.append(attempted)
        updated.resolution_status = ResolutionStatus.UNRESOLVED
        return updated

    def add_suggestions(
        self, state: ConversationState, suggestions: Iterable[str]
    ) -> tuple[ConversationState, list[str]]:
        updated = state.model_copy(deep=True)
        known = {normalize_step(step) for step in updated.troubleshooting_steps}
        attempted = {normalize_step(step) for step in updated.attempted_steps}
        accepted: list[str] = []

        for suggestion in suggestions:
            cleaned = " ".join(suggestion.strip().split())
            normalized = normalize_step(cleaned)
            if not normalized or normalized in known or normalized in attempted:
                continue
            updated.troubleshooting_steps.append(cleaned)
            accepted.append(cleaned)
            known.add(normalized)
        return updated, accepted

    @staticmethod
    def _find_referenced_step(
        state: ConversationState, user_message: str
    ) -> str:
        unattempted = [
            step
            for step in state.troubleshooting_steps
            if normalize_step(step)
            not in {normalize_step(item) for item in state.attempted_steps}
        ]
        if not unattempted:
            return ""

        message_tokens = set(normalize_step(user_message).split())
        for step in reversed(unattempted):
            meaningful_tokens = {
                token
                for token in normalize_step(step).split()
                if len(token) > 3 and token not in {"please", "your", "then"}
            }
            if meaningful_tokens & message_tokens:
                return step
        return unattempted[-1]

