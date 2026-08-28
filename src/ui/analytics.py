"""Analytics aggregation and optional Pandas/Plotly dashboard adapters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class AnalyticsSummary:
    """Aggregate metrics derived only from supplied support events."""

    conversation_count: int
    event_count: int
    resolution_rate: Optional[float]
    average_response_time_ms: Optional[float]
    average_customer_satisfaction: Optional[float]
    common_issues: dict[str, int]
    escalation_trends: dict[str, int]
    escalation_rate: Optional[float]
    average_confidence: Optional[float]
    feedback_count: int
    positive_feedback_rate: Optional[float]
    negative_feedback_rate: Optional[float]


def summarize_events(events: Iterable[Mapping[str, Any] | Any]) -> AnalyticsSummary:
    """Calculate dashboard metrics without inventing values for missing data.

    Legacy mapping events and typed ``SupportEvent`` objects are accepted.
    Missing values never become artificial zeroes: every rate uses only rows
    that actually provide its corresponding field.
    """

    rows = [_as_mapping(event) for event in events]
    outcome_rows = [row for row in rows if _is_outcome(row)]
    resolved = [_is_resolved(row) for row in outcome_rows if _has_resolution(row)]
    response_times = [float(row["response_time_ms"]) for row in outcome_rows if row.get("response_time_ms") is not None]
    confidence = [float(row["response_confidence"]) for row in outcome_rows if row.get("response_confidence") is not None]
    satisfaction = [float(row["customer_satisfaction"]) for row in rows if row.get("customer_satisfaction") is not None]
    issues = Counter(
        str(row["issue_type"]) for row in outcome_rows if row.get("issue_type")
    )
    known_escalation = [bool(row["escalation_required"]) for row in outcome_rows if row.get("escalation_required") is not None]
    feedback = [str(row["feedback_rating"]).lower() for row in rows if row.get("feedback_rating") is not None]
    escalations: Counter[str] = Counter()
    for row in outcome_rows:
        if row.get("escalation_required"):
            bucket = _date_bucket(row.get("timestamp"))
            if bucket:
                escalations[bucket] += 1

    conversation_ids = {str(row["conversation_id"]) for row in rows if row.get("conversation_id")}
    return AnalyticsSummary(
        # Legacy callers supplied generic event mappings without conversation
        # IDs, where each row historically represented one interaction.
        conversation_count=len(conversation_ids) if conversation_ids else len(rows),
        event_count=len(rows),
        resolution_rate=(sum(resolved) / len(resolved)) if resolved else None,
        average_response_time_ms=fmean(response_times) if response_times else None,
        average_customer_satisfaction=fmean(satisfaction) if satisfaction else None,
        common_issues=dict(issues.most_common()),
        escalation_trends=dict(sorted(escalations.items())),
        escalation_rate=(sum(known_escalation) / len(known_escalation)) if known_escalation else None,
        average_confidence=fmean(confidence) if confidence else None,
        feedback_count=len(feedback),
        positive_feedback_rate=(feedback.count("positive") / len(feedback)) if feedback else None,
        negative_feedback_rate=(feedback.count("negative") / len(feedback)) if feedback else None,
    )


def to_dataframe(events: Iterable[Mapping[str, Any] | Any]) -> Any:
    """Create a Pandas dataframe when the optional analytics dependency exists."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pandas is required for dataframe analytics.") from exc
    return pd.DataFrame([_as_mapping(event) for event in events])


def escalation_trend_chart(summary: AnalyticsSummary) -> Any:
    """Build a Plotly escalation trend chart from real summary data."""

    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Plotly is required for analytics charts.") from exc
    return px.line(
        x=list(summary.escalation_trends),
        y=list(summary.escalation_trends.values()),
        labels={"x": "Date", "y": "Escalations"},
        title="Escalation trend",
    )


def _date_bucket(value: object) -> Optional[str]:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None
    return None


def _as_mapping(event: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        return event
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise TypeError("analytics events must be mappings or Pydantic models")


def _has_resolution(row: Mapping[str, Any]) -> bool:
    return row.get("resolved") is not None or row.get("resolution_status") is not None


def _is_resolved(row: Mapping[str, Any]) -> bool:
    if row.get("resolved") is not None:
        return bool(row["resolved"])
    status = row.get("resolution_status")
    value = getattr(status, "value", status)
    return str(value).upper() == "RESOLVED"


def _is_outcome(row: Mapping[str, Any]) -> bool:
    event_type = getattr(row.get("event_type"), "value", row.get("event_type"))
    return event_type != "feedback"
