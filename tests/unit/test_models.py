"""Contract tests for shared Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import (
    ArticleReference,
    ChatResponse,
    Citation,
    ConversationMessage,
    ConversationState,
    KnowledgeDocument,
    MessageRole,
    ProactiveAnalysis,
    ResolutionStatus,
    RetrievalResult,
    Sentiment,
    Severity,
)


def test_knowledge_document_can_be_instantiated() -> None:
    document = KnowledgeDocument(
        id="doc-1",
        content="Restart the supported client.",
        source="vpn-guide",
        product="Oracle VPN",
        severity=Severity.MEDIUM,
        metadata={"tags": ["vpn", "connectivity"], "authoritative": True},
    )

    assert document.id == "doc-1"
    assert document.metadata["authoritative"] is True


def test_retrieval_result_can_be_instantiated() -> None:
    result = RetrievalResult(
        document_id="doc-1",
        content="Restart the supported client.",
        score=0.91,
        metadata={"source_type": "product_documentation"},
    )

    assert result.score == pytest.approx(0.91)


def test_conversation_state_can_be_instantiated() -> None:
    state = ConversationState(
        conversation_id="conversation-1",
        user_id="user-1",
        messages=[
            ConversationMessage(
                role=MessageRole.USER,
                content="My VPN is not connecting.",
            )
        ],
        product="Oracle VPN",
        resolution_status=ResolutionStatus.UNDERSTANDING,
        troubleshooting_steps=["Check the client version"],
        attempted_steps=[],
        turn_count=1,
    )

    assert state.messages[0].role is MessageRole.USER
    assert state.turn_count == 1


@pytest.mark.parametrize("status", list(ResolutionStatus))
def test_all_conversation_status_values_are_valid(status: ResolutionStatus) -> None:
    state = ConversationState(
        conversation_id="conversation-1",
        resolution_status=status.value,
    )

    assert state.resolution_status is status


def test_proactive_analysis_can_be_instantiated() -> None:
    analysis = ProactiveAnalysis(
        sentiment=Sentiment.NEGATIVE,
        frustration_score=0.75,
        escalation_required=True,
        reason="repeated_failed_troubleshooting",
        recommended_articles=[ArticleReference(article_id="article-7")],
    )

    assert analysis.escalation_required is True
    assert analysis.recommended_articles[0].article_id == "article-7"


def test_chat_response_can_be_instantiated() -> None:
    response = ChatResponse(
        message="Try the documented connection check.",
        citations=[Citation(source="vpn-guide", document_id="doc-1")],
        suggested_actions=["Confirm whether the connection succeeds"],
        confidence=0.88,
        related_articles=[ArticleReference(article_id="article-7")],
    )

    assert response.citations[0].document_id == "doc-1"
    assert response.escalation_required is False


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (KnowledgeDocument, {"id": "doc-1", "source": "guide"}),
        (RetrievalResult, {"content": "content", "score": 0.8}),
        (ConversationState, {"conversation_id": "conversation-1", "turn_count": -1}),
        (ProactiveAnalysis, {"frustration_score": 1.1}),
        (ChatResponse, {"message": "response", "confidence": -0.1}),
    ],
)
def test_invalid_required_or_bounded_values_are_rejected(model, values) -> None:
    with pytest.raises(ValidationError):
        model(**values)


def test_invalid_resolution_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ConversationState(
            conversation_id="conversation-1",
            resolution_status="CLOSED",
        )

