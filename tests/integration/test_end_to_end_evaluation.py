"""Credential-free end-to-end evaluation through the real application graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from src.analytics import FeedbackRating, InMemoryAnalyticsEventSink, SupportEvent
from src.api import ChatApplicationService
from src.app import create_application
from src.config import Settings
from src.conversation import GeneratedResponse, GenerationContext, InMemoryConversationMemory
from src.evaluation import EvaluationCase, EvaluationDataset, EvaluationRunner, RecordingRetriever
from src.ingestion import IngestionRecord, SourceType
from src.models import (
    ConversationState,
    KnowledgeDocument,
    ProactiveAnalysis,
    ResolutionStatus,
    RetrievalResult,
)
from src.proactive import ConversationMemoryHistoryProvider, ProactiveSupportService


class IndexedScenarioRetriever:
    def __init__(self, *, score: float = 0.9) -> None:
        self.score = score
        self.documents: list[KnowledgeDocument] = []
        self.search_calls: list[dict[str, object]] = []

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        self.documents.extend(document.model_copy(deep=True) for document in documents)

    def search(
        self, *, query: str, filters: Mapping[str, str], top_k: int
    ) -> Sequence[RetrievalResult]:
        self.search_calls.append(
            {"query": query, "filters": dict(filters), "top_k": top_k}
        )
        return [
            RetrievalResult(
                document_id=document.id,
                content=document.content,
                score=self.score,
                metadata={**document.metadata, "source": document.source},
            )
            for document in self.documents[:top_k]
        ]


class ScenarioLLM:
    def __init__(self) -> None:
        self.contexts: list[GenerationContext] = []

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        self.contexts.append(context)
        return GeneratedResponse(
            message="Refresh the authentication token and confirm whether it works.",
            suggested_actions=("Refresh the authentication token",),
            confidence=0.85,
        )


class NeutralProactive:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ConversationState]] = []

    def analyze(
        self, *, message: str, conversation: ConversationState
    ) -> ProactiveAnalysis:
        self.calls.append((message, conversation.model_copy(deep=True)))
        return ProactiveAnalysis()


class FailingAnalyticsSink:
    def record(self, _event: SupportEvent) -> None:
        raise RuntimeError("analytics unavailable")


def build_evaluation(
    *,
    score: float = 0.9,
    analytics_sink: object | None = None,
    conversation_id_factory=None,
    index_knowledge: bool = True,
    use_real_proactive: bool = True,
):
    sink = analytics_sink if analytics_sink is not None else InMemoryAnalyticsEventSink()
    memory = InMemoryConversationMemory()
    indexed = IndexedScenarioRetriever(score=score)
    recording = RecordingRetriever(indexed)
    llm = ScenarioLLM()
    services = create_application(
        settings=Settings(),
        retrieval_service=recording,
        llm_service=llm,
        proactive_service=None if use_real_proactive else NeutralProactive(),
        memory=memory,
        analytics_sink=sink,
    )
    documents = (
        services.knowledge_indexer.ingest_records_and_index(
            [
                IngestionRecord(
                    source="official-vpn-guide",
                    source_type=SourceType.TROUBLESHOOTING_GUIDE,
                    content=(
                        "Oracle VPN version 5.2 authentication failed. "
                        "Refresh the authentication token."
                    ),
                )
            ]
        )
        if index_knowledge
        else []
    )
    chat_service = ChatApplicationService(
        services.conversation_engine, analytics_sink=sink
    )
    runner = EvaluationRunner(
        chat_service,
        state_reader=services.conversation_engine,
        analytics_source=sink if hasattr(sink, "events") else None,
        retrieval_observer=recording,
        user_id="evaluation-user",
        conversation_id_factory=conversation_id_factory,
    )
    return services, runner, indexed, recording, llm, sink, documents


def evaluation_case(**updates: object) -> EvaluationCase:
    values: dict[str, object] = {
        "case_id": "supported",
        "issue_type": "authentication",
        "category": "authentication_login",
        "user_message": "My Oracle VPN version 5.2 reports authentication failed.",
        "expected_resolution_status": ResolutionStatus.AWAITING_CONFIRMATION,
        "expected_escalation": False,
        "expected_keywords": ["token"],
    }
    values.update(updates)
    return EvaluationCase(**values)


def test_successful_flow_uses_ingestion_retrieval_llm_memory_and_analytics() -> None:
    services, runner, indexed, _, llm, sink, documents = build_evaluation()
    case = evaluation_case(expected_source_ids=[documents[0].id])

    report = runner.run(EvaluationDataset(cases=[case]))
    result = report.cases[0]

    assert result.overall_pass is True
    assert result.actual_resolution_status is ResolutionStatus.AWAITING_CONFIRMATION
    assert result.retrieval_score == 0.9
    assert result.citation_count == 1
    assert result.suggested_action_count == 1
    assert report.metrics.expected_keyword_match_rate == 1.0
    assert report.metrics.expected_source_match_rate == 1.0
    assert report.metrics.total_cases == 1
    assert report.metrics.pass_rate == 1.0
    assert report.metrics.resolution_rate == 0.0
    assert report.metrics.escalation_rate == 0.0
    assert report.metrics.average_confidence == 0.85
    assert report.metrics.median_confidence == 0.85
    assert report.metrics.average_response_time_ms is not None
    assert report.metrics.citation_rate == 1.0
    assert report.metrics.suggested_action_rate == 1.0
    assert report.metrics.retrieval_hit_rate == 1.0
    assert report.metrics.average_retrieval_score == 0.9
    assert report.metrics.resolution_status_accuracy == 1.0
    assert report.metrics.escalation_decision_accuracy == 1.0
    assert len(indexed.search_calls) == 1
    assert len(llm.contexts) == 1
    assert isinstance(services.proactive_service, ProactiveSupportService)
    assert llm.contexts[0].retrieved_knowledge[0].document_id == documents[0].id
    assert (
        llm.contexts[0].proactive_analysis.recommended_articles[0].article_id
        == documents[0].id
    )
    assert services.memory.load(next(iter({event.conversation_id for event in sink.events()})))
    assert len(sink.events()) == 1


def test_low_confidence_retrieval_escalates_and_is_correlated_in_metrics() -> None:
    _, runner, _, _, llm, sink, _ = build_evaluation(score=0.3)
    case = evaluation_case(
        case_id="low-confidence",
        expected_resolution_status=ResolutionStatus.ESCALATED,
        expected_escalation=True,
        expected_keywords=[],
    )

    report = runner.run(EvaluationDataset(cases=[case]))
    result = report.cases[0]

    assert result.actual_escalation is True
    assert result.retrieval_score == 0.3
    assert report.metrics.cases_below_confidence_threshold == 1
    assert not llm.contexts
    assert sink.events()[-1].escalation_required is True


def test_no_retrieval_evidence_remains_distinct_from_unavailable_telemetry() -> None:
    _, runner, indexed, _, llm, _, _ = build_evaluation(index_knowledge=False)
    case = evaluation_case(
        case_id="no-evidence",
        expected_resolution_status=ResolutionStatus.ESCALATED,
        expected_escalation=True,
        expected_keywords=[],
    )

    report = runner.run(EvaluationDataset(cases=[case]))
    result = report.cases[0]

    assert len(indexed.search_calls) == 1
    assert result.retrieval_hit is False
    assert result.retrieval_score is None
    assert report.metrics.retrieval_hit_rate == 0.0
    assert report.metrics.cases_with_no_retrieval_evidence == 1
    assert report.metrics.average_retrieval_score is None
    assert not llm.contexts


def test_explicit_escalation_bypasses_retrieval_and_updates_state_and_analytics() -> None:
    services, runner, indexed, _, _, sink, _ = build_evaluation(
        use_real_proactive=False
    )
    case = evaluation_case(
        case_id="human",
        issue_type=None,
        category="escalation_request",
        user_message="Please connect me to a human support agent.",
        expected_resolution_status=ResolutionStatus.ESCALATED,
        expected_escalation=True,
        expected_keywords=["human"],
    )

    report = runner.run(EvaluationDataset(cases=[case]))
    result = report.cases[0]

    assert result.overall_pass is True
    assert result.retrieval_hit is None
    assert not indexed.search_calls
    event = sink.events()[-1]
    assert event.resolution_status is ResolutionStatus.ESCALATED
    assert event.escalation_required is True
    assert services.memory.load(event.conversation_id).resolution_status is ResolutionStatus.ESCALATED


def test_resolution_confirmation_and_feedback_are_consumed_from_real_events() -> None:
    _, runner, _, _, _, sink, _ = build_evaluation()
    case = evaluation_case(
        case_id="resolved",
        follow_up_messages=["Yes, it works now."],
        expected_resolution_status=ResolutionStatus.RESOLVED,
        expected_escalation=False,
        expected_keywords=[],
        feedback_rating=FeedbackRating.POSITIVE,
    )

    report = runner.run(EvaluationDataset(cases=[case]))
    result = report.cases[0]

    assert result.actual_resolution_status is ResolutionStatus.RESOLVED
    assert result.feedback_rating is FeedbackRating.POSITIVE
    assert report.metrics.resolved_cases == 1
    assert report.metrics.resolution_rate == 1.0
    assert [event.event_type.value for event in sink.events()] == [
        "chat_outcome",
        "chat_outcome",
        "feedback",
    ]


def test_multiple_conversations_preserve_user_scoped_history_without_cross_user_data() -> None:
    services, runner, _, _, _, _, _ = build_evaluation(
        conversation_id_factory=lambda case: case.case_id
    )
    dataset = EvaluationDataset(
        cases=[
            evaluation_case(case_id="history-one"),
            evaluation_case(case_id="history-two"),
        ]
    )
    runner.run(dataset)
    history = ConversationMemoryHistoryProvider(services.memory)

    own = history.customer_history(
        "VPN help",
        ConversationState(
            conversation_id="current",
            user_id="evaluation-user",
            product="Oracle VPN",
            issue_type="authentication",
        ),
    )
    other = history.customer_history(
        "VPN help",
        ConversationState(
            conversation_id="other-current",
            user_id="another-user",
            product="Oracle VPN",
            issue_type="authentication",
        ),
    )
    assert {reference.article_id for reference in own} == {"history-one", "history-two"}
    assert other == []


def test_analytics_failure_does_not_break_chat_or_evaluation() -> None:
    _, runner, _, _, _, _, _ = build_evaluation(analytics_sink=FailingAnalyticsSink())
    report = runner.run(EvaluationDataset(cases=[evaluation_case()]))

    assert report.cases[0].overall_pass is True
    assert report.cases[0].response_time_ms is None
    assert report.metrics.average_response_time_ms is None
