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
    resolution_rate: Optional[float]
    average_response_time_ms: Optional[float]
    average_customer_satisfaction: Optional[float]
    common_issues: dict[str, int]
    escalation_trends: dict[str, int]


def summarize_events(events: Iterable[Mapping[str, Any]]) -> AnalyticsSummary:
    """Calculate dashboard metrics without inventing values for missing data.

    Recognized optional event keys are ``resolved``, ``response_time_ms``,
    ``customer_satisfaction``, ``issue_type``, ``escalation_required``, and
    ``timestamp`` (an ISO-8601 string, date, or datetime).
    """

    rows = list(events)
    resolved = [bool(row["resolved"]) for row in rows if row.get("resolved") is not None]
    response_times = [float(row["response_time_ms"]) for row in rows if row.get("response_time_ms") is not None]
    satisfaction = [float(row["customer_satisfaction"]) for row in rows if row.get("customer_satisfaction") is not None]
    issues = Counter(str(row["issue_type"]) for row in rows if row.get("issue_type"))
    escalations: Counter[str] = Counter()
    for row in rows:
        if row.get("escalation_required"):
            bucket = _date_bucket(row.get("timestamp"))
            if bucket:
                escalations[bucket] += 1

    return AnalyticsSummary(
        conversation_count=len(rows),
        resolution_rate=(sum(resolved) / len(resolved)) if resolved else None,
        average_response_time_ms=fmean(response_times) if response_times else None,
        average_customer_satisfaction=fmean(satisfaction) if satisfaction else None,
        common_issues=dict(issues.most_common()),
        escalation_trends=dict(sorted(escalations.items())),
    )


def to_dataframe(events: Iterable[Mapping[str, Any]]) -> Any:
    """Create a Pandas dataframe when the optional analytics dependency exists."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pandas is required for dataframe analytics.") from exc
    return pd.DataFrame(list(events))


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
