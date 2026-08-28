"""Dependency ports for optional analytics collection."""

from __future__ import annotations

from typing import Protocol, Sequence

from .events import SupportEvent


class AnalyticsEventSink(Protocol):
    """Accept support events without affecting the support workflow."""

    def record(self, event: SupportEvent) -> None:
        """Record an immutable snapshot of an event."""


class AnalyticsEventReader(Protocol):
    """Optional read capability used by the local analytics view."""

    def events(self) -> Sequence[SupportEvent]:
        """Return event snapshots in their original order."""


class NoOpAnalyticsEventSink:
    """Default sink used when analytics collection is not configured."""

    def record(self, event: SupportEvent) -> None:
        del event

    def events(self) -> Sequence[SupportEvent]:
        return ()
