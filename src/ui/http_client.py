"""Concrete HTTP implementation of the Streamlit ``ChatApiClient`` port."""

from __future__ import annotations

from typing import Optional

import httpx

from src.analytics import FeedbackRating, SupportEvent
from src.api.schemas import ChatRequest, FeedbackRequest
from src.models import ChatResponse


class ChatApiClientError(RuntimeError):
    """Safe UI-facing error for unavailable or invalid API responses."""


class HttpChatApiClient:
    """Call the FastAPI chat endpoint without coupling UI code to services."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = normalized_base_url
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None
        self.last_resolution_status: Optional[str] = None

    def chat(
        self,
        *,
        conversation_id: str,
        user_message: str,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        request = ChatRequest(
            conversation_id=conversation_id,
            user_message=user_message,
            user_id=user_id,
        )
        try:
            response = self._client.post(
                f"{self._base_url}/chat",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatApiClientError("The support API timed out. Please try again.") from exc
        except httpx.HTTPStatusError as exc:
            raise ChatApiClientError("The support API could not process the request.") from exc
        except httpx.HTTPError as exc:
            raise ChatApiClientError("The support API is unavailable. Please try again.") from exc
        try:
            chat_response = ChatResponse.model_validate(response.json())
        except (TypeError, ValueError) as exc:
            raise ChatApiClientError("The support API returned an invalid response.") from exc
        self.last_resolution_status = response.headers.get("X-Resolution-Status")
        return chat_response

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def submit_feedback(
        self,
        *,
        conversation_id: str,
        rating: FeedbackRating,
        comment: Optional[str] = None,
    ) -> None:
        request = FeedbackRequest(
            conversation_id=conversation_id,
            rating=rating.value,
            comment=comment,
        )
        try:
            response = self._client.post(
                f"{self._base_url}/feedback",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatApiClientError("The support API timed out. Please try again.") from exc
        except httpx.HTTPStatusError as exc:
            raise ChatApiClientError("The support API could not process the request.") from exc
        except httpx.HTTPError as exc:
            raise ChatApiClientError("The support API is unavailable. Please try again.") from exc

    def analytics_events(self) -> list[SupportEvent]:
        try:
            response = self._client.get(f"{self._base_url}/analytics/events")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ChatApiClientError("The support API timed out. Please try again.") from exc
        except httpx.HTTPStatusError as exc:
            raise ChatApiClientError("The support API could not process the request.") from exc
        except httpx.HTTPError as exc:
            raise ChatApiClientError("The support API is unavailable. Please try again.") from exc
        try:
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("analytics response must be a list")
            return [SupportEvent.model_validate(event) for event in payload]
        except (TypeError, ValueError) as exc:
            raise ChatApiClientError("The support API returned an invalid response.") from exc
