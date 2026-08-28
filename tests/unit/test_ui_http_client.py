"""Tests for the Streamlit-facing HTTP client without a live API server."""

from __future__ import annotations

import json

import httpx
import pytest

from src.ui import ChatApiClientError, HttpChatApiClient


def make_client(handler, *, base_url: str = "http://api.example.test/") -> HttpChatApiClient:
    return HttpChatApiClient(
        base_url,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_http_client_posts_existing_request_payload_and_parses_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"X-Resolution-Status": "AWAITING_CONFIRMATION"},
            json={
                "message": "Use the documented reset.",
                "citations": [{"source": "VPN guide", "document_id": "vpn-guide"}],
                "suggested_actions": ["Reset the token"],
                "escalation_required": False,
                "confidence": 0.8,
                "related_articles": [],
            },
        )

    api_client = make_client(handler)
    response = api_client.chat(
        conversation_id="conversation-1", user_id="user-1", user_message="VPN login fails"
    )

    assert captured == {
        "url": "http://api.example.test/chat",
        "payload": {
            "conversation_id": "conversation-1",
            "user_id": "user-1",
            "user_message": "VPN login fails",
        },
    }
    assert response.message == "Use the documented reset."
    assert response.citations[0].document_id == "vpn-guide"
    assert api_client.last_resolution_status == "AWAITING_CONFIRMATION"


def test_http_client_handles_http_and_network_failures_safely() -> None:
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, json={"detail": "internal infrastructure"})

    with pytest.raises(ChatApiClientError, match="could not process") as error:
        make_client(error_handler).chat(conversation_id="c", user_message="help")
    assert "infrastructure" not in str(error.value)

    def network_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("endpoint details")

    with pytest.raises(ChatApiClientError, match="unavailable") as network_error:
        make_client(network_handler).chat(conversation_id="c", user_message="help")
    assert "endpoint details" not in str(network_error.value)


def test_http_client_handles_timeouts_and_invalid_responses() -> None:
    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout detail")

    with pytest.raises(ChatApiClientError, match="timed out"):
        make_client(timeout_handler).chat(conversation_id="c", user_message="help")

    def invalid_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": ""})

    with pytest.raises(ChatApiClientError, match="invalid response"):
        make_client(invalid_handler).chat(conversation_id="c", user_message="help")


def test_http_client_validates_base_url_and_preserves_configurability() -> None:
    with pytest.raises(ValueError, match="base_url"):
        HttpChatApiClient("   ")
    with pytest.raises(ValueError, match="timeout_seconds"):
        HttpChatApiClient("http://api.example.test", timeout_seconds=0)
