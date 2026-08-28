"""Presentation and analytics components for the customer-support UI."""

from .analytics import AnalyticsSummary, escalation_trend_chart, summarize_events, to_dataframe
from .app import ChatApiClient, render_chatbot
from .http_client import ChatApiClientError, HttpChatApiClient
from .presentation import render_streamlit_response, response_view

__all__ = [
    "AnalyticsSummary",
    "ChatApiClient",
    "ChatApiClientError",
    "HttpChatApiClient",
    "escalation_trend_chart",
    "render_chatbot",
    "render_streamlit_response",
    "response_view",
    "summarize_events",
    "to_dataframe",
]
