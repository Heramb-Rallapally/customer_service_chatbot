"""Typed contracts for repeatable end-to-end support evaluation."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.analytics import FeedbackRating
from src.models import ResolutionStatus


class EvaluationCase(BaseModel):
    """One labelled support scenario evaluated through the application boundary."""

    case_id: str = Field(min_length=1)
    issue_type: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    user_message: str = Field(min_length=1)
    follow_up_messages: list[str] = Field(default_factory=list)
    expected_resolution_status: ResolutionStatus
    expected_escalation: bool
    expected_keywords: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    feedback_rating: Optional[FeedbackRating] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "case_id", "issue_type", "category", "difficulty", "user_message"
    )
    @classmethod
    def normalize_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator(
        "follow_up_messages", "expected_keywords", "expected_source_ids"
    )
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.strip().split()) for value in values]
        if any(not value for value in normalized):
            raise ValueError("list values must not be blank")
        return normalized


class EvaluationDataset(BaseModel):
    """Versioned collection of uniquely identified evaluation cases."""

    version: str = Field(default="1.0", min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case_id values must be unique")
        return self


class EvaluationCaseResult(BaseModel):
    """Privacy-bounded outcome for one evaluated support scenario."""

    case_id: str
    issue_type: Optional[str]
    category: Optional[str]
    actual_resolution_status: Optional[ResolutionStatus]
    expected_resolution_status: ResolutionStatus
    resolution_match: bool
    actual_escalation: bool
    expected_escalation: bool
    escalation_match: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    response_time_ms: Optional[float] = Field(default=None, ge=0.0)
    citation_count: int = Field(ge=0)
    suggested_action_count: int = Field(ge=0)
    retrieval_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retrieval_hit: Optional[bool] = None
    keyword_match: Optional[bool] = None
    source_match: Optional[bool] = None
    feedback_rating: Optional[FeedbackRating] = None
    overall_pass: bool
    failure_reasons: list[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Deterministic aggregate metrics with unavailable values represented by None."""

    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    resolved_cases: Optional[int] = Field(default=None, ge=0)
    unresolved_cases: Optional[int] = Field(default=None, ge=0)
    escalated_cases: int = Field(ge=0)
    pass_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resolution_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    escalation_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    median_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_response_time_ms: Optional[float] = Field(default=None, ge=0.0)
    citation_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suggested_action_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retrieval_hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_retrieval_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cases_with_no_retrieval_evidence: Optional[int] = Field(default=None, ge=0)
    cases_below_confidence_threshold: Optional[int] = Field(default=None, ge=0)
    resolution_status_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    escalation_decision_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    expected_keyword_match_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    expected_source_match_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retrieval_confidence_threshold: float = Field(ge=0.0, le=1.0)


class BreakdownMetric(BaseModel):
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    pass_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EvaluationBreakdown(BaseModel):
    issue_type: dict[str, BreakdownMetric] = Field(default_factory=dict)
    resolution_status: dict[str, BreakdownMetric] = Field(default_factory=dict)
    escalation_status: dict[str, BreakdownMetric] = Field(default_factory=dict)
    retrieval_quality: dict[str, BreakdownMetric] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    cases_evaluated: int = Field(ge=0)
    pass_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resolution_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    escalation_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_response_time_ms: Optional[float] = Field(default=None, ge=0.0)


class EvaluationReport(BaseModel):
    """Machine-readable report generated exclusively from observed outcomes."""

    summary: EvaluationSummary
    metrics: EvaluationMetrics
    breakdown: EvaluationBreakdown
    cases: list[EvaluationCaseResult]
