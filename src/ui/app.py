"""Streamlit chat shell that delegates all support behavior to the API."""

from __future__ import annotations

from typing import Any, Optional, Protocol
from uuid import uuid4

from src.config import get_settings
from src.analytics import FeedbackRating, SupportEvent
from src.models import ChatResponse

from src.ui.http_client import HttpChatApiClient
from src.ui.analytics import escalation_trend_chart, summarize_events
from src.ui.presentation import render_streamlit_response


class ChatApiClient(Protocol):
    """UI-facing client contract for the deployed chat API."""

    def chat(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        """Submit one turn to the backend API/application service."""

    def submit_feedback(
        self,
        *,
        conversation_id: str,
        rating: FeedbackRating,
        comment: Optional[str] = None,
    ) -> None:
        """Submit authenticated helpfulness feedback through the API."""

    def analytics_events(self) -> list[SupportEvent]:
        """Return the current authenticated user's analytics event snapshots."""


def render_chatbot(
    st: Any,
    client: ChatApiClient,
    *,
    conversation_id: str,
    user_id: Optional[str] = None,
) -> None:
    """Render a minimal Streamlit chat experience using an injected API client."""

    st.title("Customer Support")
    history = st.session_state.setdefault("support_messages", [])
    for role, content in history:
        with st.chat_message(role):
            st.write(content)

    message = st.chat_input("Describe the issue you need help with")
    if not message:
        return

    with st.chat_message("user"):
        st.write(message)
    history.append(("user", message))

    try:
        response = client.chat(
            conversation_id=conversation_id,
            user_message=message,
            user_id=user_id,
        )
    except Exception:
        st.error("Unable to reach support right now. Please try again shortly.")
        return

    with st.chat_message("assistant"):
        render_streamlit_response(st, response)
        resolution_status = getattr(client, "last_resolution_status", None)
        if resolution_status:
            st.caption(f"Resolution status: {resolution_status}")
    history.append(("assistant", response.message))
    _render_feedback(st, client, conversation_id, len(history))


def _render_feedback(st: Any, client: ChatApiClient, conversation_id: str, turn_key: int) -> None:
    """Render a minimal API-backed helpfulness control for the latest turn."""

    st.caption("Was this response helpful?")
    positive, negative = st.columns(2)
    try:
        if positive.button("👍", key=f"feedback-positive-{turn_key}"):
            client.submit_feedback(
                conversation_id=conversation_id, rating=FeedbackRating.POSITIVE
            )
            st.success("Thanks for the feedback.")
        if negative.button("👎", key=f"feedback-negative-{turn_key}"):
            client.submit_feedback(
                conversation_id=conversation_id, rating=FeedbackRating.NEGATIVE
            )
            st.info("Thanks for the feedback. A support specialist can review it.")
    except Exception:
        st.warning("Feedback could not be submitted right now.")


def render_analytics(st: Any, events: list[SupportEvent]) -> None:
    """Render an intentionally small view from real API-provided event data."""

    summary = summarize_events(events)
    st.subheader("Your support activity")
    columns = st.columns(4)
    columns[0].metric("Total interactions", summary.event_count)
    columns[1].metric("Resolution rate", _format_percent(summary.resolution_rate))
    columns[2].metric("Escalation rate", _format_percent(summary.escalation_rate))
    columns[3].metric("Feedback count", summary.feedback_count)
    columns = st.columns(2)
    columns[0].metric("Average response time", _format_ms(summary.average_response_time_ms))
    columns[1].metric("Average confidence", _format_percent(summary.average_confidence))
    st.metric("Positive feedback rate", _format_percent(summary.positive_feedback_rate))
    if summary.escalation_trends:
        st.plotly_chart(escalation_trend_chart(summary), use_container_width=True)


def _format_percent(value: Optional[float]) -> str:
    return "Not available" if value is None else f"{value:.0%}"


def _format_ms(value: Optional[float]) -> str:
    return "Not available" if value is None else f"{value:.0f} ms"


def main() -> None:
    """Run the Streamlit chat shell against the configured FastAPI service."""

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - dependency is runtime-only.
        raise RuntimeError("Streamlit is required to run the UI.") from exc

    settings = get_settings()
    client = st.session_state.get("chat_api_client")
    if client is None:
        client = HttpChatApiClient(settings.api_base_url)
        st.session_state["chat_api_client"] = client
    conversation_id = st.session_state.setdefault("conversation_id", str(uuid4()))
    user_id = st.sidebar.text_input("User ID (optional)") or None
    st.sidebar.caption("Escalation and resolution status are shown with each response.")
    if st.sidebar.checkbox("Show my support analytics"):
        try:
            render_analytics(st, client.analytics_events())
        except Exception:
            st.sidebar.warning("Analytics are unavailable right now.")
    st.caption(f"Conversation: {conversation_id}")
    render_chatbot(st, client, conversation_id=conversation_id, user_id=user_id)


if __name__ == "__main__":  # pragma: no cover - launched by Streamlit.
    main()
