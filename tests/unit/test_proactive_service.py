"""Exercise proactive support through local fakes without external dependencies.

These tests protect shared output compatibility, deterministic escalation, and graceful
provider failure behavior. Future engineers should add scenarios here before changing
scoring thresholds or provider-boundary semantics.
"""

from __future__ import annotations

import pytest

from src.models import ArticleReference, ConversationState, ResolutionStatus, Sentiment, Severity
from src.proactive import OciSentimentAnalyzer, ProactiveSupportService


class FixedSentiment:
    """Return a supplied sentiment to isolate service behavior in unit tests."""

    def __init__(self, sentiment: Sentiment) -> None:
        self.sentiment = sentiment

    def analyze(self, message: str) -> Sentiment:
        return self.sentiment


class EvidenceProvider:
    """Supply only evidence references; it does not perform retrieval or persistence."""

    def related_articles(self, message: str, conversation: ConversationState | None):
        return [{"id": "guide-1", "title": "VPN guide"}]

    def similar_issues(self, message: str, conversation: ConversationState | None):
        return [ArticleReference(article_id="issue-2")]

    def historical_solutions(self, message: str, conversation: ConversationState | None):
        return [{"article_id": "ticket-3", "source": "reviewed_ticket"}]

    def customer_history(self, message: str, conversation: ConversationState | None):
        return [{"document_id": "history-4"}]


class UnsupportedDetector:
    """Represent an injected, evidence-backed unsupported issue signal."""

    def is_unsupported(self, message: str, conversation: ConversationState | None) -> bool:
        return True


class FailingProvider:
    """Simulate an unavailable dependency without exposing exception details."""

    def related_articles(self, message: str, conversation: ConversationState | None):
        raise RuntimeError("sensitive provider failure")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Thanks, that worked.", Sentiment.POSITIVE),
        ("Please check my account.", Sentiment.NEUTRAL),
        ("This is broken and frustrating.", Sentiment.NEGATIVE),
        ("", Sentiment.UNKNOWN),
    ],
)
def test_fallback_sentiment_paths(message: str, expected: Sentiment) -> None:
    assert ProactiveSupportService().analyze(message).sentiment is expected


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_blank_messages_are_safe_and_unknown(message: str) -> None:
    analysis = ProactiveSupportService().analyze(message)
    assert analysis.sentiment is Sentiment.UNKNOWN
    assert analysis.frustration_score == 0.0
    assert analysis.escalation_required is False


def test_frustration_is_bounded_and_negative_language_increases_it() -> None:
    service = ProactiveSupportService()
    neutral = service.analyze("Can you check this?")
    negative = service.analyze("This is broken and I am frustrated.")
    assert 0.0 <= neutral.frustration_score <= 1.0
    assert 0.0 <= negative.frustration_score <= 1.0
    assert negative.frustration_score > neutral.frustration_score


def test_human_request_and_repeated_failure_escalate() -> None:
    service = ProactiveSupportService()
    assert service.analyze("I need a live agent now.").reason == "human_support_requested"
    state = ConversationState(conversation_id="c-1", attempted_steps=["restart"])
    assert service.analyze("I already tried it and it didn't work.", state).reason == "repeated_failed_troubleshooting"


@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.CRITICAL])
def test_high_severity_escalates(severity: Severity) -> None:
    state = ConversationState(conversation_id="c-1", severity=severity)
    analysis = ProactiveSupportService().analyze("Please help", state)
    assert analysis.escalation_required is True
    assert analysis.reason == "high_severity"


def test_unsupported_signal_escalates() -> None:
    analysis = ProactiveSupportService(unsupported_issue_detector=UnsupportedDetector()).analyze("Please help")
    assert analysis.reason == "unsupported_issue"


def test_attempted_steps_and_turns_increase_frustration() -> None:
    baseline = ProactiveSupportService().analyze("Please help")
    state = ConversationState(
        conversation_id="c-1",
        attempted_steps=["restart", "reinstall"],
        turn_count=5,
        resolution_status=ResolutionStatus.UNRESOLVED,
    )
    assert ProactiveSupportService().analyze("Please help", state).frustration_score > baseline.frustration_score


def test_high_severity_increases_frustration() -> None:
    baseline = ProactiveSupportService().analyze("Please help")
    high_severity = ConversationState(conversation_id="c-1", severity=Severity.HIGH)
    assert ProactiveSupportService().analyze("Please help", high_severity).frustration_score > baseline.frustration_score


def test_provider_evidence_is_mapped_with_source_labels() -> None:
    analysis = ProactiveSupportService(
        recommendation_provider=EvidenceProvider(), history_provider=EvidenceProvider()
    ).analyze("VPN help")
    assert [(article.article_id, article.source) for article in analysis.recommended_articles] == [
        ("guide-1", "knowledge_article"),
        ("issue-2", "similar_issue"),
        ("ticket-3", "reviewed_ticket"),
        ("history-4", "customer_history"),
    ]


def test_provider_failure_is_safe_and_does_not_create_recommendations() -> None:
    analysis = ProactiveSupportService(recommendation_provider=FailingProvider()).analyze("VPN help")
    assert analysis.recommended_articles == []


def test_oci_adapter_is_mockable_and_requires_no_credentials() -> None:
    assert OciSentimentAnalyzer(lambda message: {"sentiment": "negative"}).analyze("help") is Sentiment.NEGATIVE
    assert OciSentimentAnalyzer(lambda message: {"sentiment": " neutral "}).analyze("help") is Sentiment.NEUTRAL
    assert OciSentimentAnalyzer(lambda message: {"sentiment": "invalid"}).analyze("help") is Sentiment.UNKNOWN
    assert OciSentimentAnalyzer(lambda message: (_ for _ in ()).throw(RuntimeError())).analyze("help") is Sentiment.UNKNOWN
