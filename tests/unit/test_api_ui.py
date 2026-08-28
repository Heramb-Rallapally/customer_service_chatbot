"""Member 5 unit tests using an injected mock conversation service."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.api import ChatApplicationService, ChatRequest
from src.models import ArticleReference, ChatResponse, Citation
from src.ui import response_view, summarize_events


class StubConversationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def handle_message(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_id: str | None = None,
    ) -> ChatResponse:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "user_message": user_message,
                "user_id": user_id,
            }
        )
        return ChatResponse(
            message="Use the documented connection check.", confidence=0.9
        )


def test_chat_service_delegates_to_injected_conversation_service() -> None:
    engine = StubConversationService()
    service = ChatApplicationService(engine)

    response = service.chat(
        ChatRequest(conversation_id="c-1", user_id="u-1", user_message="VPN is down")
    )

    assert response.confidence == 0.9
    assert engine.calls == [
        {"conversation_id": "c-1", "user_message": "VPN is down", "user_id": "u-1"}
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"conversation_id": " ", "user_message": "Hello"},
        {"conversation_id": "c-1", "user_message": "  "},
    ],
)
def test_chat_request_rejects_blank_required_values(values: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(**values)


def test_response_view_exposes_all_customer_facing_response_details() -> None:
    response = ChatResponse(
        message="A support specialist will help.",
        citations=[
            Citation(
                source="VPN guide", document_id="guide-1", excerpt="Connection steps"
            )
        ],
        suggested_actions=["Confirm the client version"],
        escalation_required=True,
        confidence=0.4,
        related_articles=[
            ArticleReference(article_id="article-1", title="VPN connection guide")
        ],
    )

    assert response_view(response) == {
        "message": "A support specialist will help.",
        "citations": [
            {
                "source": "VPN guide",
                "document_id": "guide-1",
                "excerpt": "Connection steps",
            }
        ],
        "suggested_actions": ["Confirm the client version"],
        "related_articles": [
            {"article_id": "article-1", "title": "VPN connection guide"}
        ],
        "escalation_required": True,
        "confidence": 0.4,
    }


def test_analytics_summarizes_provided_events_without_filling_missing_metrics() -> None:
    summary = summarize_events(
        [
            {
                "resolved": True,
                "response_time_ms": 120,
                "customer_satisfaction": 5,
                "issue_type": "connectivity",
                "escalation_required": True,
                "timestamp": datetime(2026, 8, 28, tzinfo=timezone.utc),
            },
            {
                "resolved": False,
                "response_time_ms": 80,
                "issue_type": "connectivity",
                "escalation_required": False,
            },
            {
                "issue_type": "billing",
                "escalation_required": True,
                "timestamp": "2026-08-29T10:00:00Z",
            },
        ]
    )

    assert summary.conversation_count == 3
    assert summary.resolution_rate == 0.5
    assert summary.average_response_time_ms == 100
    assert summary.average_customer_satisfaction == 5
    assert summary.common_issues == {"connectivity": 2, "billing": 1}
    assert summary.escalation_trends == {"2026-08-28": 1, "2026-08-29": 1}


def test_empty_analytics_has_no_fabricated_metrics() -> None:
    summary = summarize_events([])

    assert summary.conversation_count == 0
    assert summary.resolution_rate is None
    assert summary.average_response_time_ms is None
    assert summary.average_customer_satisfaction is None
