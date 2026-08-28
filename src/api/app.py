"""FastAPI application factory for customer-support chat."""

from __future__ import annotations

import logging
from typing import Any

from src.models import ChatResponse

from .schemas import ChatRequest
from .service import ChatApplicationService, ConversationServiceUnavailableError

logger = logging.getLogger(__name__)


def create_app(chat_service: ChatApplicationService) -> Any:
    """Create the HTTP app with injected conversation orchestration.

    FastAPI is intentionally imported here so shared modules and unit tests can
    run before the API/UI dependency set is installed.
    """

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FastAPI is required to create the API application. "
            "Install the API/UI dependencies before starting the server."
        ) from exc

    app = FastAPI(title="Customer Service Chatbot API", version="0.1.0")

    @app.post("/chat", response_model=ChatResponse)
    def post_chat(request: ChatRequest) -> ChatResponse:
        try:
            return chat_service.chat(request)
        except ConversationServiceUnavailableError as exc:
            logger.warning("Conversation service unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="The support service is temporarily unavailable. Please try again.",
            ) from exc
        except Exception:
            logger.exception("Unexpected failure while processing a chat request")
            raise HTTPException(
                status_code=500,
                detail="Unable to process the support request right now.",
            )

    return app
