"""Unit tests for Conversation Engine orchestration boundaries and decisions."""

from typing import Mapping, Optional, Sequence

import pytest

from src.conversation import (
    ConversationEngine,
    ConversationEngineOptions,
    GeneratedResponse,
    GenerationContext,
    InMemoryConversationMemory,
)
from src.models import (
    ConversationMessage,
    ConversationState,
    MessageRole,
    ProactiveAnalysis,
    ResolutionStatus,
    RetrievalResult,
    Sentiment,
)


class FakeRetriever:
    def __init__(
        self,
        results: Sequence[RetrievalResult] = (),
        error: Optional[Exception] = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls: list[dict] = []

    def search(
        self, *, query: str, filters: Mapping[str, str], top_k: int
    ) -> Sequence[RetrievalResult]:
        self.calls.append({"query": query, "filters": dict(filters), "top_k": top_k})
        if self.error:
            raise self.error
        return list(self.results)


class FakeLLM:
    def __init__(
        self,
        response: Optional[GeneratedResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.response = response or GeneratedResponse(message="Supported guidance")
        self.error = error
        self.contexts: list[GenerationContext] = []

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        self.contexts.append(context)
        if self.error:
            raise self.error
        return self.response


class FakeProactive:
    def __init__(
        self,
        analysis: Optional[ProactiveAnalysis] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.analysis = analysis or ProactiveAnalysis()
        self.error = error
        self.calls: list[tuple[str, ConversationState]] = []

    def analyze(
        self, *, message: str, conversation: ConversationState
    ) -> ProactiveAnalysis:
        self.calls.append((message, conversation))
        if self.error:
            raise self.error
        return self.analysis


def result(score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        document_id="doc-1",
        content="Refresh the authentication token.",
        score=score,
        metadata={"source": "Oracle VPN guide"},
    )


def ready_state(**updates) -> ConversationState:
    values = {
        "conversation_id": "conversation-1",
        "product": "Oracle VPN",
        "version": "5.2",
        "issue_type": "authentication",
        "issue_summary": "authentication failed",
        "resolution_status": ResolutionStatus.READY_TO_RESOLVE,
    }
    values.update(updates)
    return ConversationState(**values)


def build_engine(
    *,
    memory: Optional[InMemoryConversationMemory] = None,
    retriever: Optional[FakeRetriever] = None,
    llm: Optional[FakeLLM] = None,
    proactive: Optional[FakeProactive] = None,
    options: Optional[ConversationEngineOptions] = None,
) -> tuple[ConversationEngine, InMemoryConversationMemory, FakeRetriever, FakeLLM]:
    actual_memory = memory or InMemoryConversationMemory()
    actual_retriever = retriever or FakeRetriever([result()])
    actual_llm = llm or FakeLLM()
    engine = ConversationEngine(
        retriever=actual_retriever,
        llm_service=actual_llm,
        proactive_service=proactive,
        memory=actual_memory,
        options=options,
    )
    return engine, actual_memory, actual_retriever, actual_llm


def test_missing_information_is_requested_without_repeating_known_product() -> None:
    engine, _, retriever, _ = build_engine()

    first = engine.handle_message(
        conversation_id="conversation-1", user_message="My VPN isn't working."
    )
    second = engine.handle_message(
        conversation_id="conversation-1", user_message="Oracle VPN."
    )

    assert "product or client" in first.message
    assert "version" in second.message
    assert "product or client" not in second.message
    assert not retriever.calls
    state = engine.get_state("conversation-1")
    assert state is not None
    assert state.product == "Oracle VPN"
    assert state.turn_count == 2


def test_retriever_and_llm_receive_focused_structured_context() -> None:
    memory = InMemoryConversationMemory()
    memory.save(
        ready_state(
            messages=[
                ConversationMessage(role=MessageRole.USER, content="Older question"),
                ConversationMessage(role=MessageRole.ASSISTANT, content="Older answer"),
            ]
        )
    )
    memory.set_summary("conversation-1", "VPN authentication issue.")
    llm = FakeLLM(
        GeneratedResponse(
            message="Refresh the token, then tell me whether it works.",
            suggested_actions=("Refresh the authentication token",),
            confidence=0.8,
        )
    )
    engine, _, retriever, _ = build_engine(
        memory=memory,
        llm=llm,
        options=ConversationEngineOptions(recent_message_limit=1),
    )

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="What should I try next?"
    )

    assert retriever.calls == [
        {
            "query": "Oracle VPN 5.2 authentication failed",
            "filters": {
                "product": "Oracle VPN",
                "version": "5.2",
                "issue_type": "authentication",
            },
            "top_k": 5,
        }
    ]
    context = llm.contexts[0]
    assert context.current_user_message == "What should I try next?"
    assert context.conversation_summary == "VPN authentication issue."
    assert [message.content for message in context.recent_messages] == ["Older answer"]
    assert context.conversation.messages == []
    assert context.retrieved_knowledge[0].document_id == "doc-1"
    assert response.citations[0].source == "Oracle VPN guide"
    assert response.suggested_actions == ["Refresh the authentication token"]
    assert response.confidence == pytest.approx(0.8)


def test_medium_confidence_marks_generation_context_as_cautious() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state())
    llm = FakeLLM()
    engine, _, _, _ = build_engine(
        memory=memory,
        retriever=FakeRetriever([result(0.6)]),
        llm=llm,
    )

    engine.handle_message(conversation_id="conversation-1", user_message="Help")

    assert llm.contexts[0].cautious is True


def test_successful_resolution_requires_customer_confirmation() -> None:
    memory = InMemoryConversationMemory()
    memory.save(
        ready_state(
            troubleshooting_steps=["Refresh the authentication token"],
            resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        )
    )
    engine, _, retriever, llm = build_engine(memory=memory)

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="Yes, it works now."
    )

    state = engine.get_state("conversation-1")
    assert state is not None
    assert state.resolution_status is ResolutionStatus.RESOLVED
    assert "resolved" in response.message
    assert not retriever.calls
    assert not llm.contexts


def test_failed_step_is_tracked_and_excluded_from_next_generation() -> None:
    memory = InMemoryConversationMemory()
    memory.save(
        ready_state(
            troubleshooting_steps=["Refresh the authentication token"],
            resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        )
    )
    llm = FakeLLM(
        GeneratedResponse(
            message="Clear cached credentials and retry.",
            suggested_actions=("Clear cached credentials",),
        )
    )
    engine, _, _, _ = build_engine(memory=memory, llm=llm)

    response = engine.handle_message(
        conversation_id="conversation-1",
        user_message="I tried that and it still doesn't work.",
    )

    state = engine.get_state("conversation-1")
    assert state is not None
    assert state.attempted_steps == ["Refresh the authentication token"]
    assert "Refresh the authentication token" in llm.contexts[0].excluded_steps
    assert response.suggested_actions == ["Clear cached credentials"]
    assert state.resolution_status is ResolutionStatus.AWAITING_CONFIRMATION


def test_repeated_generated_failed_step_is_not_returned() -> None:
    memory = InMemoryConversationMemory()
    memory.save(
        ready_state(
            troubleshooting_steps=["Refresh the authentication token"],
            resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        )
    )
    llm = FakeLLM(
        GeneratedResponse(
            message="Try refreshing the token again.",
            suggested_actions=("Refresh the authentication token",),
        )
    )
    engine, _, _, _ = build_engine(memory=memory, llm=llm)

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="I already tried that."
    )

    assert response.escalation_required is True
    assert response.suggested_actions == []


def test_explicit_human_request_escalates_without_retrieval() -> None:
    engine, _, retriever, _ = build_engine()

    response = engine.handle_message(
        conversation_id="conversation-1",
        user_message="Please connect me to a human agent.",
    )

    assert response.escalation_required is True
    assert not retriever.calls
    assert engine.get_state("conversation-1").resolution_status is ResolutionStatus.ESCALATED


def test_high_frustration_signal_escalates() -> None:
    proactive = FakeProactive(
        ProactiveAnalysis(
            sentiment=Sentiment.NEGATIVE,
            frustration_score=0.9,
        )
    )
    engine, _, retriever, _ = build_engine(proactive=proactive)

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="My VPN isn't working."
    )

    assert response.escalation_required is True
    assert "frustrating" in response.message
    assert not retriever.calls


@pytest.mark.parametrize("results", [[], [result(0.2)]])
def test_empty_or_low_confidence_retrieval_uses_grounded_fallback(results) -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state())
    llm = FakeLLM()
    engine, _, _, _ = build_engine(
        memory=memory,
        retriever=FakeRetriever(results),
        llm=llm,
    )

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="What should I try?"
    )

    assert response.escalation_required is True
    assert "won't guess" in response.message
    assert not llm.contexts


def test_retrieval_failure_is_hidden_behind_safe_fallback() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state())
    engine, _, _, _ = build_engine(
        memory=memory,
        retriever=FakeRetriever(error=RuntimeError("secret infrastructure detail")),
    )

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="What should I try?"
    )

    assert response.escalation_required is True
    assert "temporarily unavailable" in response.message
    assert "secret infrastructure detail" not in response.message


def test_llm_failure_preserves_real_retrieval_citation_and_escalates() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state())
    engine, _, _, _ = build_engine(
        memory=memory,
        llm=FakeLLM(error=RuntimeError("provider failed")),
    )

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="What should I try?"
    )

    assert response.escalation_required is True
    assert response.citations[0].document_id == "doc-1"


def test_proactive_failure_does_not_block_grounded_response() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state())
    engine, _, _, llm = build_engine(
        memory=memory,
        proactive=FakeProactive(error=RuntimeError("analysis unavailable")),
    )

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="What should I try?"
    )

    assert response.message == "Supported guidance"
    assert len(llm.contexts) == 1


def test_repeated_failed_troubleshooting_escalates_before_another_search() -> None:
    memory = InMemoryConversationMemory()
    memory.save(
        ready_state(
            troubleshooting_steps=["Refresh token", "Clear credentials"],
            attempted_steps=["Refresh token"],
            resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        )
    )
    engine, _, retriever, _ = build_engine(memory=memory)

    response = engine.handle_message(
        conversation_id="conversation-1", user_message="I tried that; it didn't work."
    )

    assert response.escalation_required is True
    assert "Multiple" in response.message
    assert not retriever.calls


def test_invalid_input_and_cross_user_access_are_rejected() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ConversationState(conversation_id="conversation-1", user_id="user-1"))
    engine, _, _, _ = build_engine(memory=memory)

    with pytest.raises(ValueError, match="user_message"):
        engine.handle_message(conversation_id="conversation-1", user_message="  ")
    with pytest.raises(ValueError, match="another user"):
        engine.handle_message(
            conversation_id="conversation-1",
            user_id="user-2",
            user_message="Help",
        )


def test_existing_user_bound_conversation_requires_matching_user_id() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state(user_id="user-1"))
    engine, _, retriever, _ = build_engine(memory=memory)

    engine.handle_message(
        conversation_id="conversation-1",
        user_id="user-1",
        user_message="What should I try?",
    )

    assert len(retriever.calls) == 1

    with pytest.raises(ValueError, match="another user"):
        engine.handle_message(
            conversation_id="conversation-1",
            user_id="user-2",
            user_message="What should I try?",
        )


def test_existing_user_bound_conversation_rejects_omitted_user_id() -> None:
    memory = InMemoryConversationMemory()
    memory.save(ready_state(user_id="user-1"))
    engine, _, _, _ = build_engine(memory=memory)

    with pytest.raises(ValueError, match="another user"):
        engine.handle_message(
            conversation_id="conversation-1",
            user_message="What should I try?",
        )


def test_new_conversation_allows_and_binds_user_id() -> None:
    engine, memory, _, _ = build_engine()

    engine.handle_message(
        conversation_id="new-conversation",
        user_id="user-1",
        user_message="My VPN isn't working.",
    )

    state = memory.load("new-conversation")
    assert state is not None
    assert state.user_id == "user-1"
