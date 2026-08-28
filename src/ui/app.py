"""Streamlit chat shell that delegates all support behavior to the API."""

from __future__ import annotations

from typing import Any, Optional, Protocol
from uuid import uuid4

from src.config import get_settings
from src.models import ChatResponse

from src.ui.http_client import HttpChatApiClient
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
    st.caption(f"Conversation: {conversation_id}")
    render_chatbot(st, client, conversation_id=conversation_id, user_id=user_id)


if __name__ == "__main__":  # pragma: no cover - launched by Streamlit.
    main()
