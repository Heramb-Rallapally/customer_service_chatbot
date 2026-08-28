"""Small report fixture kept independent of OCI and Oracle infrastructure."""

from __future__ import annotations

import pytest

from src.evaluation.models import (
    EvaluationBreakdown,
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationSummary,
)
from src.models import ResolutionStatus


@pytest.fixture
def local_evaluation_report() -> EvaluationReport:
    case = EvaluationCaseResult(
        case_id="case-1",
        issue_type="authentication",
        category="authentication_login",
        actual_resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        expected_resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        resolution_match=True,
        actual_escalation=False,
        expected_escalation=False,
        escalation_match=True,
        confidence=None,
        response_time_ms=None,
        citation_count=0,
        suggested_action_count=0,
        retrieval_score=None,
        retrieval_hit=None,
        overall_pass=True,
    )
    metrics = EvaluationMetrics(
        total_cases=1,
        passed_cases=1,
        resolved_cases=0,
        unresolved_cases=1,
        escalated_cases=0,
        pass_rate=1.0,
        resolution_rate=0.0,
        escalation_rate=0.0,
        citation_rate=0.0,
        suggested_action_rate=0.0,
        resolution_status_accuracy=1.0,
        escalation_decision_accuracy=1.0,
        retrieval_confidence_threshold=0.45,
    )
    return EvaluationReport(
        summary=EvaluationSummary(
            cases_evaluated=1,
            pass_rate=1.0,
            resolution_rate=0.0,
            escalation_rate=0.0,
        ),
        metrics=metrics,
        breakdown=EvaluationBreakdown(),
        cases=[case],
    )
