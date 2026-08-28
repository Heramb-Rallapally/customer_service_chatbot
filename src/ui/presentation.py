"""UI-ready formatting of shared chat responses, with no business logic."""

from __future__ import annotations

from typing import Any

from src.models import ChatResponse


def response_view(response: ChatResponse) -> dict[str, Any]:
    """Return presentation fields while preserving the shared response values."""

    return {
        "message": response.message,
        "citations": [citation.model_dump(exclude_none=True) for citation in response.citations],
        "suggested_actions": response.suggested_actions,
        "related_articles": [article.model_dump(exclude_none=True) for article in response.related_articles],
        "escalation_required": response.escalation_required,
        "confidence": response.confidence,
    }


def render_streamlit_response(st: Any, response: ChatResponse) -> None:
    """Render a response in Streamlit when the optional UI dependency is present."""

    view = response_view(response)
    st.write(view["message"])
    if view["confidence"] is not None:
        st.caption(f"Confidence: {view['confidence']:.0%}")
    if view["escalation_required"]:
        st.warning("This issue has been marked for escalation to human support.")
    if view["citations"]:
        st.markdown("**Sources**")
        for citation in view["citations"]:
            label = citation["source"]
            if citation.get("excerpt"):
                label = f"{label}: {citation['excerpt']}"
            st.write(f"- {label}")
    if view["suggested_actions"]:
        st.markdown("**Suggested actions**")
        for action in view["suggested_actions"]:
            st.write(f"- {action}")
    if view["related_articles"]:
        st.markdown("**Related articles**")
        for article in view["related_articles"]:
            st.write(f"- {article.get('title') or article['article_id']}")
