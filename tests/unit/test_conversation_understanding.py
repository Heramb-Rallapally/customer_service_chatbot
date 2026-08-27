"""Tests for deterministic conversation understanding helpers."""

import pytest

from src.conversation.context import (
    ClarificationPlanner,
    ContextUpdater,
    RetrievalQueryBuilder,
)
from src.conversation.intent import ConversationIntent, IntentDetector
from src.models import ConversationMessage, ConversationState, MessageRole, ResolutionStatus


@pytest.mark.parametrize(
    ("message", "state", "expected"),
    [
        ("My VPN is broken", ConversationState(conversation_id="c1"), ConversationIntent.NEW_ISSUE),
        (
            "Oracle VPN",
            ConversationState(
                conversation_id="c1",
                messages=[ConversationMessage(role=MessageRole.USER, content="VPN issue")],
                resolution_status=ResolutionStatus.NEEDS_CLARIFICATION,
            ),
            ConversationIntent.CLARIFICATION_RESPONSE,
        ),
        (
            "I already tried that",
            ConversationState(conversation_id="c1"),
            ConversationIntent.TROUBLESHOOTING_RESULT,
        ),
        (
            "Please connect me to a human",
            ConversationState(conversation_id="c1"),
            ConversationIntent.ESCALATION_REQUEST,
        ),
        (
            "Can you explain that?",
            ConversationState(
                conversation_id="c1",
                messages=[ConversationMessage(role=MessageRole.USER, content="Question")],
                resolution_status=ResolutionStatus.UNDERSTANDING,
            ),
            ConversationIntent.FOLLOW_UP,
        ),
        (
            "I have another issue",
            ConversationState(
                conversation_id="c1",
                messages=[ConversationMessage(role=MessageRole.USER, content="Old issue")],
                resolution_status=ResolutionStatus.RESOLVED,
            ),
            ConversationIntent.NEW_REQUEST,
        ),
    ],
)
def test_intent_handling(message, state, expected) -> None:
    assert IntentDetector().detect(message, state) is expected


def test_resolution_confirmation_requires_prior_troubleshooting() -> None:
    state = ConversationState(
        conversation_id="c1",
        troubleshooting_steps=["Refresh the token"],
        resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
    )

    assert (
        IntentDetector().detect("Yes, it works now", state)
        is ConversationIntent.RESOLUTION_CONFIRMATION
    )


def test_context_is_retained_and_enriched_across_clarifications() -> None:
    updater = ContextUpdater()
    state = ConversationState(
        conversation_id="c1",
        issue_summary="My VPN isn't working",
        resolution_status=ResolutionStatus.NEEDS_CLARIFICATION,
    )

    state = updater.update(state, "Oracle VPN.", ConversationIntent.CLARIFICATION_RESPONSE)
    state = updater.update(state, "Version 5.2.", ConversationIntent.CLARIFICATION_RESPONSE)
    state = updater.update(state, "Authentication failed.", ConversationIntent.CLARIFICATION_RESPONSE)

    assert state.product == "Oracle VPN"
    assert state.version == "5.2"
    assert state.issue_type == "authentication"
    assert state.issue_summary == "authentication failed"


def test_clarification_planner_does_not_ask_for_known_information() -> None:
    planner = ClarificationPlanner()
    state = ConversationState(conversation_id="c1", product="Oracle VPN")

    clarification = planner.next_question(state)

    assert clarification is not None
    assert clarification.field == "version"


def test_retrieval_query_uses_structured_context_and_filters() -> None:
    state = ConversationState(
        conversation_id="c1",
        product="Oracle VPN",
        version="5.2",
        issue_type="authentication",
        issue_summary="authentication failed",
    )

    query, filters = RetrievalQueryBuilder().build(state, "It says authentication failed")

    assert query == "Oracle VPN 5.2 authentication failed"
    assert filters == {
        "product": "Oracle VPN",
        "version": "5.2",
        "issue_type": "authentication",
    }
