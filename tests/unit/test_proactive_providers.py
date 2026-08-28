"""Credential-free tests for retrieval and memory proactive providers."""

from __future__ import annotations

from typing import Sequence

from src.conversation import ConversationEngine, GeneratedResponse, InMemoryConversationMemory
from src.models import ConversationState, ResolutionStatus, RetrievalResult
from src.proactive import (
    ConversationMemoryHistoryProvider,
    ProactiveSupportService,
    RetrievalEvidenceProvider,
    SupportLevel,
)


class FakeRetriever:
    def __init__(self, results: Sequence[object] = (), error: Exception | None = None) -> None:
        self.results = results
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search(self, *, query: str, filters: dict[str, str], top_k: int) -> Sequence[object]:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        if self.error is not None:
            raise self.error
        return self.results


def result(
    document_id: str = "vpn-guide",
    score: float = 0.9,
    metadata: dict[str, object] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        content="Use the documented VPN reset.",
        score=score,
        metadata=(
            {"source": "VPN guide", "title": "VPN reset"}
            if metadata is None
            else metadata
        ),
    )


def state(*, user_id: str | None = "user-a", conversation_id: str = "current") -> ConversationState:
    return ConversationState(
        conversation_id=conversation_id,
        user_id=user_id,
        product="Oracle VPN",
        version="5.2",
        issue_type="authentication",
    )


def test_retrieval_evidence_provider_preserves_retrieval_reference_metadata_once() -> None:
    retriever = FakeRetriever(
        [
            result(),
            result(
                "ticket-1",
                metadata={"source": "Resolved ticket", "source_type": "historical_ticket"},
            ),
        ]
    )
    provider = RetrievalEvidenceProvider(retriever)

    assert provider.assess_support("VPN login fails", state()) is SupportLevel.SUPPORTED
    assert provider.is_unsupported("VPN login fails", state()) is False
    related = provider.related_articles("VPN login fails", state())
    similar = provider.similar_issues("VPN login fails", state())

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["filters"] == {
        "product": "Oracle VPN",
        "version": "5.2",
        "issue_type": "authentication",
    }
    assert [(item.article_id, item.title, item.source) for item in related] == [
        ("vpn-guide", "VPN reset", "VPN guide"),
        ("ticket-1", None, "Resolved ticket"),
    ]
    assert [(item.article_id, item.source) for item in similar] == [
        ("ticket-1", "Resolved ticket")
    ]


def test_retrieval_evidence_distinguishes_weak_missing_and_unavailable_results() -> None:
    weak = RetrievalEvidenceProvider(FakeRetriever([result(score=0.4)]))
    missing = RetrievalEvidenceProvider(FakeRetriever())
    unavailable = RetrievalEvidenceProvider(FakeRetriever(error=RuntimeError("OCI endpoint")))

    assert weak.assess_support("VPN help", state()) is SupportLevel.POTENTIALLY_SUPPORTED
    assert weak.is_unsupported("VPN help", state()) is False
    assert missing.assess_support("VPN help", state()) is SupportLevel.UNSUPPORTED
    assert missing.is_unsupported("VPN help", state()) is True
    assert unavailable.assess_support("VPN help", state()) is SupportLevel.UNAVAILABLE
    assert unavailable.is_unsupported("VPN help", state()) is False


def test_retrieval_provider_ignores_non_result_and_malformed_reference_evidence() -> None:
    provider = RetrievalEvidenceProvider(FakeRetriever([object(), result(metadata={})]))

    references = provider.related_articles("VPN help", state())

    assert [reference.article_id for reference in references] == ["vpn-guide"]
    assert references[0].title is None
    assert references[0].source is None


def test_proactive_service_reuses_one_retrieval_for_detection_and_recommendations() -> None:
    retriever = FakeRetriever([result()])
    evidence = RetrievalEvidenceProvider(retriever)
    service = ProactiveSupportService(
        recommendation_provider=evidence,
        unsupported_issue_detector=evidence,
    )

    analysis = service.analyze("VPN help", state())

    assert analysis.escalation_required is False
    assert [article.article_id for article in analysis.recommended_articles] == ["vpn-guide"]
    assert len(retriever.calls) == 1


def test_conversation_engine_reuses_matching_proactive_retrieval_evidence() -> None:
    retriever = FakeRetriever([result()])
    evidence = RetrievalEvidenceProvider(retriever)

    class FakeLLM:
        def generate(self, _context: object) -> GeneratedResponse:
            return GeneratedResponse(message="Use the documented reset.")

    memory = InMemoryConversationMemory()
    ready = state()
    ready.resolution_status = ResolutionStatus.READY_TO_RESOLVE
    memory.save(ready)
    engine = ConversationEngine(
        retriever=retriever,
        llm_service=FakeLLM(),
        proactive_service=ProactiveSupportService(
            recommendation_provider=evidence,
            unsupported_issue_detector=evidence,
        ),
        memory=memory,
    )

    response = engine.handle_message(
        conversation_id="current", user_id="user-a", user_message="Can I reset the token?"
    )

    assert response.message == "Use the documented reset."
    assert len(retriever.calls) == 1


def test_history_provider_returns_only_same_authenticated_users_prior_state() -> None:
    memory = InMemoryConversationMemory()
    resolved = state(conversation_id="user-a-resolved")
    resolved.resolution_status = ResolutionStatus.RESOLVED
    memory.save(resolved)
    memory.save(state(conversation_id="user-a-open"))
    memory.save(state(user_id="user-b", conversation_id="user-b-private"))
    provider = ConversationMemoryHistoryProvider(memory)

    history = provider.customer_history("VPN help", state())
    solutions = provider.historical_solutions("VPN help", state())

    assert {reference.article_id for reference in history} == {
        "user-a-resolved",
        "user-a-open",
    }
    assert [(reference.article_id, reference.source) for reference in solutions] == [
        ("user-a-resolved", "historical_solution")
    ]


def test_history_provider_handles_missing_or_failed_memory_safely() -> None:
    class FailingMemory:
        def list_for_user(self, *args: object, **kwargs: object) -> list[ConversationState]:
            raise RuntimeError("database credentials")

    provider = ConversationMemoryHistoryProvider(FailingMemory())  # type: ignore[arg-type]

    assert provider.customer_history("VPN help", state(user_id=None)) == []
    assert provider.customer_history("VPN help", state()) == []
