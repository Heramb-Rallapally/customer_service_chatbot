"""End-to-end evaluation runner over the existing chat application contract."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, median
from typing import Callable, Optional, Protocol, Sequence
from uuid import uuid4

from src.analytics import (
    SupportEvaluationRecord,
    SupportEvent,
    to_evaluation_records,
)
from src.api import ChatApplicationService, ChatRequest
from src.models import ChatResponse, ConversationState, ResolutionStatus

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
from .retrieval import RetrievalObservation, RetrievalObservationSource


class ConversationStateReader(Protocol):
    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Return the state persisted by the normal conversation pipeline."""


class AnalyticsEventSource(Protocol):
    def events(self) -> Sequence[SupportEvent]:
        """Return snapshots of already-recorded Step 7 events."""


class EvaluationRunner:
    """Run labelled cases through ChatApplicationService without alternate orchestration."""

    def __init__(
        self,
        chat_service: ChatApplicationService,
        *,
        state_reader: ConversationStateReader,
        analytics_source: Optional[AnalyticsEventSource] = None,
        retrieval_observer: Optional[RetrievalObservationSource] = None,
        user_id: str = "evaluation-user",
        retrieval_confidence_threshold: float = 0.45,
        conversation_id_factory: Optional[Callable[[EvaluationCase], str]] = None,
    ) -> None:
        if not user_id.strip():
            raise ValueError("user_id must not be blank")
        if not 0.0 <= retrieval_confidence_threshold <= 1.0:
            raise ValueError("retrieval_confidence_threshold must be between 0 and 1")
        self._chat_service = chat_service
        self._state_reader = state_reader
        self._analytics_source = analytics_source
        self._retrieval_observer = retrieval_observer
        self._user_id = user_id.strip()
        self._retrieval_confidence_threshold = retrieval_confidence_threshold
        self._conversation_id_factory = conversation_id_factory or (
            lambda case: f"evaluation-{case.case_id}-{uuid4()}"
        )

    def run(self, dataset: EvaluationDataset) -> EvaluationReport:
        results = [self._run_case(case) for case in dataset.cases]
        metrics = _calculate_metrics(results, self._retrieval_confidence_threshold)
        return EvaluationReport(
            summary=EvaluationSummary(
                cases_evaluated=metrics.total_cases,
                pass_rate=metrics.pass_rate,
                resolution_rate=metrics.resolution_rate,
                escalation_rate=metrics.escalation_rate,
                average_confidence=metrics.average_confidence,
                average_response_time_ms=metrics.average_response_time_ms,
            ),
            metrics=metrics,
            breakdown=_calculate_breakdown(results, self._retrieval_confidence_threshold),
            cases=results,
        )

    def _run_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        conversation_id = self._conversation_id_factory(case)
        retrieval_checkpoint = (
            self._retrieval_observer.checkpoint()
            if self._retrieval_observer is not None
            else None
        )
        response: Optional[ChatResponse] = None
        for message in [case.user_message, *case.follow_up_messages]:
            response = self._chat_service.chat(
                ChatRequest(
                    conversation_id=conversation_id,
                    user_message=message,
                    user_id=self._user_id,
                )
            )
        if response is None:  # EvaluationCase validation makes this unreachable.
            raise RuntimeError("evaluation case contains no messages")
        if case.feedback_rating is not None:
            self._chat_service.record_feedback(
                conversation_id=conversation_id,
                user_id=self._user_id,
                rating=case.feedback_rating,
            )

        state = self._state_reader.get_state(conversation_id)
        analytics_record = self._analytics_record(conversation_id)
        observations = (
            self._retrieval_observer.observations_since(retrieval_checkpoint)
            if self._retrieval_observer is not None and retrieval_checkpoint is not None
            else ()
        )
        return _case_result(
            case,
            response=response,
            state=state,
            analytics_record=analytics_record,
            retrieval_observations=observations,
        )

    def _analytics_record(
        self, conversation_id: str
    ) -> Optional[SupportEvaluationRecord]:
        if self._analytics_source is None:
            return None
        try:
            user_events = [
                event
                for event in self._analytics_source.events()
                if event.conversation_id == conversation_id
                and event.user_id == self._user_id
            ]
            matches = [
                record
                for record in to_evaluation_records(user_events)
                if record.conversation_id == conversation_id
            ]
        except Exception:
            return None
        return matches[-1] if matches else None


def _case_result(
    case: EvaluationCase,
    *,
    response: ChatResponse,
    state: Optional[ConversationState],
    analytics_record: Optional[SupportEvaluationRecord],
    retrieval_observations: Sequence[RetrievalObservation],
) -> EvaluationCaseResult:
    actual_status = (
        state.resolution_status
        if state is not None
        else _recorded_resolution_status(analytics_record)
    )
    actual_escalation = response.escalation_required
    confidence = (
        response.confidence
        if response.confidence is not None
        else analytics_record.response_confidence
        if analytics_record is not None
        else None
    )
    response_time_ms = (
        analytics_record.response_time_ms if analytics_record is not None else None
    )

    observed_results = [
        result
        for observation in retrieval_observations
        for result in observation.results
    ]
    retrieval_hit = bool(observed_results) if retrieval_observations else None
    retrieval_score = max((result.score for result in observed_results), default=None)

    keyword_match = None
    if case.expected_keywords:
        normalized_response = response.message.casefold()
        keyword_match = all(
            keyword.casefold() in normalized_response
            for keyword in case.expected_keywords
        )
    source_match = None
    if case.expected_source_ids:
        actual_sources = _response_source_identifiers(response)
        source_match = all(
            expected.casefold() in actual_sources
            for expected in case.expected_source_ids
        )

    resolution_match = actual_status is case.expected_resolution_status
    escalation_match = actual_escalation is case.expected_escalation
    failure_reasons: list[str] = []
    if not resolution_match:
        failure_reasons.append("resolution_status_mismatch")
    if not escalation_match:
        failure_reasons.append("escalation_decision_mismatch")
    if keyword_match is False:
        failure_reasons.append("expected_keywords_missing")
    if source_match is False:
        failure_reasons.append("expected_sources_missing")

    return EvaluationCaseResult(
        case_id=case.case_id,
        issue_type=case.issue_type or (state.issue_type if state is not None else None),
        category=case.category,
        actual_resolution_status=actual_status,
        expected_resolution_status=case.expected_resolution_status,
        resolution_match=resolution_match,
        actual_escalation=actual_escalation,
        expected_escalation=case.expected_escalation,
        escalation_match=escalation_match,
        confidence=confidence,
        response_time_ms=response_time_ms,
        citation_count=len(response.citations),
        suggested_action_count=len(response.suggested_actions),
        retrieval_score=retrieval_score,
        retrieval_hit=retrieval_hit,
        keyword_match=keyword_match,
        source_match=source_match,
        feedback_rating=(
            analytics_record.feedback_rating if analytics_record is not None else None
        ),
        overall_pass=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def _recorded_resolution_status(
    record: Optional[SupportEvaluationRecord],
) -> Optional[ResolutionStatus]:
    if record is None or record.resolution_status is None:
        return None
    try:
        return ResolutionStatus(record.resolution_status)
    except ValueError:
        return None


def _response_source_identifiers(response: ChatResponse) -> set[str]:
    identifiers: set[str] = set()
    for citation in response.citations:
        identifiers.add(citation.source.casefold())
        if citation.document_id:
            identifiers.add(citation.document_id.casefold())
    for article in response.related_articles:
        identifiers.add(article.article_id.casefold())
        if article.source:
            identifiers.add(article.source.casefold())
        if article.title:
            identifiers.add(article.title.casefold())
    return identifiers


def _calculate_metrics(
    results: Sequence[EvaluationCaseResult], threshold: float
) -> EvaluationMetrics:
    total = len(results)
    passed = sum(result.overall_pass for result in results)
    resolved = sum(
        result.actual_resolution_status is ResolutionStatus.RESOLVED
        for result in results
    )
    known_resolution = sum(
        result.actual_resolution_status is not None for result in results
    )
    escalated = sum(result.actual_escalation for result in results)
    unresolved = sum(
        result.actual_resolution_status is not None
        and result.actual_resolution_status is not ResolutionStatus.RESOLVED
        and not result.actual_escalation
        for result in results
    )
    confidences = [result.confidence for result in results if result.confidence is not None]
    response_times = [
        result.response_time_ms
        for result in results
        if result.response_time_ms is not None
    ]
    observed_retrieval = [
        result.retrieval_hit for result in results if result.retrieval_hit is not None
    ]
    retrieval_scores = [
        result.retrieval_score
        for result in results
        if result.retrieval_score is not None
    ]
    keyword_matches = [
        result.keyword_match for result in results if result.keyword_match is not None
    ]
    source_matches = [
        result.source_match for result in results if result.source_match is not None
    ]
    return EvaluationMetrics(
        total_cases=total,
        passed_cases=passed,
        resolved_cases=resolved if known_resolution else None,
        unresolved_cases=unresolved if known_resolution else None,
        escalated_cases=escalated,
        pass_rate=_rate(passed, total),
        resolution_rate=_rate(resolved, known_resolution),
        escalation_rate=_rate(escalated, total),
        average_confidence=fmean(confidences) if confidences else None,
        median_confidence=median(confidences) if confidences else None,
        average_response_time_ms=fmean(response_times) if response_times else None,
        citation_rate=_rate(sum(result.citation_count > 0 for result in results), total),
        suggested_action_rate=_rate(
            sum(result.suggested_action_count > 0 for result in results), total
        ),
        retrieval_hit_rate=(
            sum(observed_retrieval) / len(observed_retrieval)
            if observed_retrieval
            else None
        ),
        average_retrieval_score=(
            fmean(retrieval_scores) if retrieval_scores else None
        ),
        cases_with_no_retrieval_evidence=(
            sum(not hit for hit in observed_retrieval)
            if observed_retrieval
            else None
        ),
        cases_below_confidence_threshold=(
            sum(score < threshold for score in retrieval_scores)
            if retrieval_scores
            else None
        ),
        resolution_status_accuracy=_rate(
            sum(result.resolution_match for result in results), total
        ),
        escalation_decision_accuracy=_rate(
            sum(result.escalation_match for result in results), total
        ),
        expected_keyword_match_rate=(
            sum(keyword_matches) / len(keyword_matches) if keyword_matches else None
        ),
        expected_source_match_rate=(
            sum(source_matches) / len(source_matches) if source_matches else None
        ),
        retrieval_confidence_threshold=threshold,
    )


def _calculate_breakdown(
    results: Sequence[EvaluationCaseResult], threshold: float
) -> EvaluationBreakdown:
    return EvaluationBreakdown(
        issue_type=_breakdown(
            results, lambda result: result.issue_type or "unavailable"
        ),
        resolution_status=_breakdown(
            results,
            lambda result: (
                result.actual_resolution_status.value
                if result.actual_resolution_status is not None
                else "unavailable"
            ),
        ),
        escalation_status=_breakdown(
            results,
            lambda result: "escalated" if result.actual_escalation else "not_escalated",
        ),
        retrieval_quality=_breakdown(
            results, lambda result: _retrieval_quality(result, threshold)
        ),
    )


def _breakdown(
    results: Sequence[EvaluationCaseResult],
    key: Callable[[EvaluationCaseResult], str],
) -> dict[str, BreakdownMetric]:
    groups: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {
        name: BreakdownMetric(
            total_cases=len(group),
            passed_cases=sum(result.overall_pass for result in group),
            pass_rate=_rate(sum(result.overall_pass for result in group), len(group)),
        )
        for name, group in sorted(groups.items())
    }


def _retrieval_quality(result: EvaluationCaseResult, threshold: float) -> str:
    if result.retrieval_hit is None:
        return "unavailable"
    if result.retrieval_hit is False:
        return "no_evidence"
    if result.retrieval_score is None:
        return "score_unavailable"
    return "below_threshold" if result.retrieval_score < threshold else "at_or_above_threshold"


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None
