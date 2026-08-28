"""Public end-to-end evaluation contracts and runners."""

from .dataset import EvaluationDatasetError, load_dataset
from .models import (
    BreakdownMetric,
    EvaluationBreakdown,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationSummary,
)
from .reporting import render_console_report, report_json
from .retrieval import RecordingRetriever, RetrievalObservation, RetrievalObservationSource
from .runner import AnalyticsEventSource, ConversationStateReader, EvaluationRunner

__all__ = [
    "AnalyticsEventSource",
    "BreakdownMetric",
    "ConversationStateReader",
    "EvaluationBreakdown",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationDataset",
    "EvaluationDatasetError",
    "EvaluationMetrics",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationSummary",
    "RecordingRetriever",
    "RetrievalObservation",
    "RetrievalObservationSource",
    "load_dataset",
    "render_console_report",
    "report_json",
]
