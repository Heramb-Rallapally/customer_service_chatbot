"""Deterministic mocked end-to-end multi-turn Conversation Engine scenario."""

from typing import Mapping, Sequence

from src.conversation import (
    ConversationEngine,
    GeneratedResponse,
    GenerationContext,
    InMemoryConversationMemory,
)
from src.models import ConversationState, ProactiveAnalysis, ResolutionStatus, RetrievalResult


class ScenarioRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def search(
        self, *, query: str, filters: Mapping[str, str], top_k: int
    ) -> Sequence[RetrievalResult]:
        self.calls.append((query, dict(filters)))
        return [
            RetrievalResult(
                document_id="vpn-auth-guide",
                content="Refresh the token. If that fails, clear cached credentials.",
                score=0.93,
                metadata={"source": "Oracle VPN authentication guide"},
            )
        ]


class ScenarioLLM:
    def __init__(self) -> None:
        self.contexts: list[GenerationContext] = []

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        self.contexts.append(context)
        if not context.excluded_steps:
            return GeneratedResponse(
                message="Refresh your authentication token, then tell me if you can connect.",
                suggested_actions=("Refresh the authentication token",),
                confidence=0.9,
            )
        return GeneratedResponse(
            message="Clear cached credentials, reconnect, and tell me if it works.",
            suggested_actions=("Clear cached credentials",),
            confidence=0.88,
        )


class NeutralProactive:
    def analyze(
        self, *, message: str, conversation: ConversationState
    ) -> ProactiveAnalysis:
        return ProactiveAnalysis()


def test_six_turn_vpn_resolution_flow() -> None:
    memory = InMemoryConversationMemory()
    retriever = ScenarioRetriever()
    llm = ScenarioLLM()
    engine = ConversationEngine(
        retriever=retriever,
        llm_service=llm,
        proactive_service=NeutralProactive(),
        memory=memory,
    )

    turn_1 = engine.handle_message(
        conversation_id="vpn-case", user_message="My VPN isn't working."
    )
    assert "product or client" in turn_1.message

    turn_2 = engine.handle_message(
        conversation_id="vpn-case", user_message="Oracle VPN."
    )
    assert "version" in turn_2.message

    turn_3 = engine.handle_message(
        conversation_id="vpn-case", user_message="Version 5.2."
    )
    assert "error message" in turn_3.message

    turn_4 = engine.handle_message(
        conversation_id="vpn-case", user_message="Authentication failed."
    )
    assert turn_4.suggested_actions == ["Refresh the authentication token"]
    assert turn_4.citations[0].document_id == "vpn-auth-guide"

    turn_5 = engine.handle_message(
        conversation_id="vpn-case",
        user_message="I tried that and it still doesn't work.",
    )
    assert turn_5.suggested_actions == ["Clear cached credentials"]
    assert turn_5.escalation_required is False
    assert llm.contexts[-1].excluded_steps == ("Refresh the authentication token",)

    turn_6 = engine.handle_message(
        conversation_id="vpn-case", user_message="Yes, it works now."
    )
    assert "resolved" in turn_6.message

    state = engine.get_state("vpn-case")
    assert state is not None
    assert state.product == "Oracle VPN"
    assert state.version == "5.2"
    assert state.issue_type == "authentication"
    assert state.issue_summary == "authentication failed"
    assert state.attempted_steps == ["Refresh the authentication token"]
    assert state.troubleshooting_steps == [
        "Refresh the authentication token",
        "Clear cached credentials",
    ]
    assert state.turn_count == 6
    assert state.resolution_status is ResolutionStatus.RESOLVED
    assert [query for query, _ in retriever.calls] == [
        "Oracle VPN 5.2 authentication failed",
        "Oracle VPN 5.2 authentication failed",
    ]

