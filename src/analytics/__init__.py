"""Optional support analytics contracts, sinks, and evaluation helpers."""

from .evaluation import SupportEvaluationRecord, to_evaluation_records
from .events import FeedbackRating, SupportEvent, SupportEventType
from .interfaces import AnalyticsEventReader, AnalyticsEventSink, NoOpAnalyticsEventSink
from .memory import InMemoryAnalyticsEventSink

__all__ = [
    "AnalyticsEventReader",
    "AnalyticsEventSink",
    "FeedbackRating",
    "InMemoryAnalyticsEventSink",
    "NoOpAnalyticsEventSink",
    "SupportEvaluationRecord",
    "SupportEvent",
    "SupportEventType",
    "to_evaluation_records",
]
