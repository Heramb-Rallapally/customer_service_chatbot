"""Thread-safe, process-local analytics storage for development and tests."""

from __future__ import annotations

from threading import RLock
from typing import Sequence

from .events import SupportEvent


class InMemoryAnalyticsEventSink:
    """Store defensive event copies in insertion order.

    This adapter is intentionally process-local. Production analytics storage is
    deferred; the sink remains optional so an unavailable analytics destination
    never interrupts chat handling.
    """

    def __init__(self) -> None:
        self._events: list[SupportEvent] = []
        self._lock = RLock()

    def record(self, event: SupportEvent) -> None:
        snapshot = event.model_copy(deep=True)
        with self._lock:
            self._events.append(snapshot)

    def events(self) -> Sequence[SupportEvent]:
        with self._lock:
            return tuple(event.model_copy(deep=True) for event in self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
