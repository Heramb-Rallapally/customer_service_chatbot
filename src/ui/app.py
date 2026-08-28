"""Streamlit chat shell that delegates all support behavior to the API."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from src.models import ChatResponse

from .presentation import render_streamlit_response


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
    history.append(("assistant", response.message))
