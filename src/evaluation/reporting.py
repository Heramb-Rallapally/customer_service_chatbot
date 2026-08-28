"""Human- and machine-readable evaluation report rendering."""

from __future__ import annotations

import json

from .models import EvaluationReport


def report_json(report: EvaluationReport, *, indent: int = 2) -> str:
    """Serialize the stable report contract without non-JSON values."""

    return json.dumps(report.model_dump(mode="json"), indent=indent, sort_keys=True)


def render_console_report(report: EvaluationReport) -> str:
    """Render only observed metrics and labelled comparisons."""

    summary = report.summary
    lines = [
        "Customer Support Evaluation",
        "===========================",
        f"Cases evaluated: {summary.cases_evaluated}",
        f"Pass rate: {_percent(summary.pass_rate)}",
        f"Resolution rate: {_percent(summary.resolution_rate)}",
        f"Escalation rate: {_percent(summary.escalation_rate)}",
        f"Average confidence: {_number(summary.average_confidence)}",
        f"Average response time: {_milliseconds(summary.average_response_time_ms)}",
        "",
        "Breakdown",
        "---------",
    ]
    for label, groups in (
        ("Issue type", report.breakdown.issue_type),
        ("Resolution status", report.breakdown.resolution_status),
        ("Escalation", report.breakdown.escalation_status),
        ("Retrieval quality", report.breakdown.retrieval_quality),
    ):
        lines.append(f"{label}:")
        for name, metric in groups.items():
            lines.append(
                f"  {name}: {metric.passed_cases}/{metric.total_cases} passed "
                f"({_percent(metric.pass_rate)})"
            )

    failed = [result for result in report.cases if not result.overall_pass]
    lines.extend(["", "Failed cases", "------------"])
    if not failed:
        lines.append("None")
    for result in failed:
        actual_status = (
            result.actual_resolution_status.value
            if result.actual_resolution_status is not None
            else "unavailable"
        )
        lines.append(
            f"{result.case_id}: expected status={result.expected_resolution_status.value}, "
            f"actual status={actual_status}, expected escalation={result.expected_escalation}, "
            f"actual escalation={result.actual_escalation}; "
            f"reasons={', '.join(result.failure_reasons)}"
        )

    lines.extend(["", "Observations", "------------"])
    observations = _observations(report)
    lines.extend(observations or ["No deterministic warning patterns were observed."])
    return "\n".join(lines)


def _observations(report: EvaluationReport) -> list[str]:
    observations: list[str] = []
    no_evidence = report.metrics.cases_with_no_retrieval_evidence
    if no_evidence:
        observations.append(f"{no_evidence} evaluated cases had no retrieval evidence.")
    low_confidence_escalations = sum(
        result.actual_escalation
        and result.retrieval_score is not None
        and result.retrieval_score < report.metrics.retrieval_confidence_threshold
        for result in report.cases
    )
    if low_confidence_escalations:
        observations.append(
            f"{low_confidence_escalations} low-retrieval-confidence cases escalated."
        )
    missing_citations = sum(
        result.retrieval_hit is True and result.citation_count == 0
        for result in report.cases
    )
    if missing_citations:
        observations.append(
            f"{missing_citations} cases with retrieval evidence returned no citations."
        )
    return observations


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _milliseconds(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1f} ms"
