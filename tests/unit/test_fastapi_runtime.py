"""Credential-free FastAPI boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import (
    AuthenticatedIdentity,
    ChatApplicationService,
    ConversationServiceUnavailableError,
    DevelopmentIdentityProvider,
    RequestStateIdentityProvider,
    create_app,
    create_runtime_app,
)
from src.analytics import InMemoryAnalyticsEventSink, SupportEvent, SupportEventType
from src.app import ApplicationConfigurationError, ApplicationInitializationError
from src.conversation import (
    ConversationEngine,
    ConversationOwnershipError,
    GeneratedResponse,
    InMemoryConversationMemory,
)
from src.models import ChatResponse, Citation, ConversationState, ProactiveAnalysis, ResolutionStatus


class ConversationDouble:
    def __init__(self, response: ChatResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str | None]] = []

    def handle_message(
        self, *, conversation_id: str, user_message: str, user_id: str | None = None
    ) -> ChatResponse:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "user_id": user_id,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def chat_client(response: ChatResponse | Exception) -> tuple[TestClient, ConversationDouble]:
    conversation = ConversationDouble(response)
    return (
        TestClient(
            create_app(
                ChatApplicationService(conversation),
                identity_provider=DevelopmentIdentityProvider("user-1"),
            )
        ),
        conversation,
    )


def test_health_is_process_only_and_returns_ok() -> None:
    client, _ = chat_client(ChatResponse(message="unused"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_uses_existing_request_and_response_contracts() -> None:
    expected = ChatResponse(
        message="Use the documented reset.",
        citations=[Citation(source="VPN guide", document_id="vpn-guide")],
        suggested_actions=["Reset the token"],
        escalation_required=False,
        confidence=0.8,
    )
    client, conversation = chat_client(expected)

    response = client.post(
        "/chat",
        json={
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "user_message": "VPN login fails",
        },
    )

    assert response.status_code == 200
    assert ChatResponse.model_validate(response.json()) == expected
    assert conversation.calls == [
        {
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "user_message": "VPN login fails",
        }
    ]


def test_chat_exposes_post_turn_resolution_status_without_changing_body() -> None:
    class StatefulConversation(ConversationDouble):
        def get_state(self, conversation_id: str) -> ConversationState:
            return ConversationState(
                conversation_id=conversation_id,
                resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
            )

    conversation = StatefulConversation(ChatResponse(message="Confirm the result."))
    client = TestClient(
        create_app(
            ChatApplicationService(conversation),
            identity_provider=DevelopmentIdentityProvider("user-1"),
        )
    )

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.headers["X-Resolution-Status"] == "AWAITING_CONFIRMATION"
    assert ChatResponse.model_validate(response.json()).message == "Confirm the result."


def test_invalid_request_is_rejected_by_pydantic() -> None:
    client, _ = chat_client(ChatResponse(message="unused"))

    response = client.post("/chat", json={"conversation_id": " ", "user_message": " "})

    assert response.status_code == 422


def test_conversation_validation_error_becomes_safe_4xx() -> None:
    client, _ = chat_client(ConversationOwnershipError("ownership details"))

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "This conversation is not available to this user."
    assert "ownership details" not in response.text


def test_downstream_failures_are_safe() -> None:
    client, _ = chat_client(RuntimeError("OCI endpoint with internal detail"))

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to process the support request right now."
    assert "OCI endpoint" not in response.text


def test_known_service_unavailability_becomes_safe_503() -> None:
    client, _ = chat_client(ConversationServiceUnavailableError("internal service detail"))

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "The support service is temporarily unavailable. Please try again."


def test_lazy_application_configuration_and_initialization_failures_are_safe_503() -> None:
    for failure in (
        ApplicationConfigurationError("Missing required configuration: ORACLE_DB_DSN"),
        ApplicationInitializationError("Oracle connection failure details"),
    ):
        def application_factory(failure=failure):
            raise failure

        client = TestClient(create_runtime_app(application_factory=application_factory))
        response = client.post(
            "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
        )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "The support service is temporarily unavailable. Please try again."
        )
        assert "ORACLE_DB_DSN" not in response.text
        assert "connection failure" not in response.text


def test_runtime_app_defers_composition_until_chat() -> None:
    calls = 0

    def application_factory():
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            conversation_engine=ConversationDouble(ChatResponse(message="Grounded response")),
            close=lambda: None,
        )

    with TestClient(create_runtime_app(application_factory=application_factory)) as client:
        assert client.get("/health").status_code == 200
        assert calls == 0
        response = client.post(
            "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Grounded response"
        assert calls == 1


class EmptyRetriever:
    def search(self, *, query: str, filters: dict[str, str], top_k: int) -> list[object]:
        return []


class UnusedLLM:
    def generate(self, context: object) -> GeneratedResponse:
        return GeneratedResponse(message="Unused")


class NeutralProactiveService:
    def analyze(self, *, message: str, conversation: ConversationState) -> ProactiveAnalysis:
        return ProactiveAnalysis()


class MutableIdentityProvider:
    def __init__(self, user_id: str) -> None:
        self.identity = AuthenticatedIdentity(user_id)

    def get_identity(self, _request: object) -> AuthenticatedIdentity:
        return self.identity


def test_http_route_enforces_real_conversation_user_ownership() -> None:
    """Exercise ownership through FastAPI, not an API-only test double."""

    engine = ConversationEngine(
        retriever=EmptyRetriever(),
        llm_service=UnusedLLM(),
        proactive_service=NeutralProactiveService(),
        memory=InMemoryConversationMemory(),
    )
    identities = MutableIdentityProvider("user-a")
    client = TestClient(
        create_app(ChatApplicationService(engine), identity_provider=identities)
    )
    initial_turn = {
        "conversation_id": "user-a-conversation",
        "user_message": "I need help with Oracle VPN version 5.2.",
    }

    assert client.post("/chat", json=initial_turn).status_code == 200
    assert client.post(
        "/chat",
        json={**initial_turn, "user_message": "It is still not working."},
    ).status_code == 200

    identities.identity = AuthenticatedIdentity("user-b")
    response = client.post(
        "/chat", json={**initial_turn, "user_message": "Show me the history."}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "This conversation is not available to this user."
    assert "another user" not in response.text

    identities.identity = AuthenticatedIdentity("user-a")
    impersonation = client.post(
        "/chat",
        json={**initial_turn, "user_id": "user-b", "user_message": "Show me the history."},
    )
    assert impersonation.status_code == 403
    assert "does not match" not in impersonation.text


def test_authenticated_identity_overrides_compatible_request_user_id() -> None:
    client, conversation = chat_client(ChatResponse(message="Grounded response"))

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 200
    assert conversation.calls[0]["user_id"] == "user-1"


def test_required_authentication_rejects_requests_without_trusted_identity() -> None:
    client = TestClient(
        create_app(
            ChatApplicationService(ConversationDouble(ChatResponse(message="unused"))),
            identity_provider=RequestStateIdentityProvider(),
        )
    )

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication is required."
    assert "identity" not in response.text.lower()


def test_request_state_identity_provider_uses_server_populated_identity() -> None:
    conversation = ConversationDouble(ChatResponse(message="Grounded response"))
    app = create_app(
        ChatApplicationService(conversation),
        identity_provider=RequestStateIdentityProvider(),
    )

    @app.middleware("http")
    async def trusted_identity_middleware(request, call_next):
        request.state.authenticated_identity = AuthenticatedIdentity("production-user")
        return await call_next(request)

    response = TestClient(app).post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 200
    assert conversation.calls[0]["user_id"] == "production-user"


def test_feedback_route_uses_authenticated_identity_and_enforces_ownership() -> None:
    class FeedbackConversation(ConversationDouble):
        def __init__(self) -> None:
            super().__init__(ChatResponse(message="Grounded response"))
            self.state = ConversationState(
                conversation_id="conversation-1",
                user_id="user-a",
                resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
            )

        def get_state(self, _conversation_id: str) -> ConversationState:
            return self.state.model_copy(deep=True)

    conversation = FeedbackConversation()
    identities = MutableIdentityProvider("user-a")
    sink = InMemoryAnalyticsEventSink()
    client = TestClient(
        create_app(
            ChatApplicationService(conversation, analytics_sink=sink),
            identity_provider=identities,
        )
    )

    accepted = client.post(
        "/feedback",
        json={"conversation_id": "conversation-1", "rating": "positive", "comment": "Useful"},
    )
    assert accepted.status_code == 200
    assert accepted.json() == {"accepted": True}
    assert sink.events()[0].user_id == "user-a"
    assert sink.events()[0].feedback_rating.value == "positive"

    identities.identity = AuthenticatedIdentity("user-b")
    denied = client.post(
        "/feedback",
        json={
            "conversation_id": "conversation-1",
            "rating": "negative",
            "user_id": "user-a",
        },
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "This conversation is not available to this user."
    assert "user-a" not in denied.text


def test_chat_returns_success_when_analytics_sink_fails() -> None:
    class FailingAnalyticsSink:
        def record(self, _event: SupportEvent) -> None:
            raise RuntimeError("analytics infrastructure detail")

    client = TestClient(
        create_app(
            ChatApplicationService(
                ConversationDouble(ChatResponse(message="Grounded response")),
                analytics_sink=FailingAnalyticsSink(),
            ),
            identity_provider=DevelopmentIdentityProvider("user-1"),
        )
    )

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Grounded response"
    assert "infrastructure" not in response.text


def test_one_accepted_chat_request_records_exactly_one_outcome_event() -> None:
    class StatefulConversation(ConversationDouble):
        def get_state(self, conversation_id: str) -> ConversationState:
            return ConversationState(conversation_id=conversation_id, user_id="user-1")

    sink = InMemoryAnalyticsEventSink()
    client = TestClient(
        create_app(
            ChatApplicationService(
                StatefulConversation(ChatResponse(message="Grounded response")),
                analytics_sink=sink,
            ),
            identity_provider=DevelopmentIdentityProvider("user-1"),
        )
    )

    response = client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "VPN help"}
    )

    assert response.status_code == 200
    events = sink.events()
    assert len(events) == 1
    assert events[0].event_type is SupportEventType.CHAT_OUTCOME


def test_feedback_requires_authenticated_identity_and_valid_payload() -> None:
    client = TestClient(
        create_app(
            ChatApplicationService(ConversationDouble(ChatResponse(message="unused"))),
            identity_provider=RequestStateIdentityProvider(),
        )
    )
    assert client.post(
        "/feedback", json={"conversation_id": "conversation-1", "rating": "positive"}
    ).status_code == 401

    authenticated = TestClient(
        create_app(
            ChatApplicationService(ConversationDouble(ChatResponse(message="unused"))),
            identity_provider=DevelopmentIdentityProvider("user-1"),
        )
    )
    assert authenticated.post(
        "/feedback", json={"conversation_id": " ", "rating": "maybe"}
    ).status_code == 422


def test_analytics_events_endpoint_is_scoped_to_authenticated_user() -> None:
    class AnalyticsConversation(ConversationDouble):
        def get_state(self, conversation_id: str) -> ConversationState:
            return ConversationState(conversation_id=conversation_id, user_id="user-a")

    identities = MutableIdentityProvider("user-a")
    sink = InMemoryAnalyticsEventSink()
    client = TestClient(
        create_app(
            ChatApplicationService(AnalyticsConversation(ChatResponse(message="response")), analytics_sink=sink),
            identity_provider=identities,
        )
    )
    assert client.post(
        "/chat", json={"conversation_id": "conversation-1", "user_message": "Help"}
    ).status_code == 200
    assert len(client.get("/analytics/events").json()) == 1
    identities.identity = AuthenticatedIdentity("user-b")
    assert client.get("/analytics/events").json() == []
