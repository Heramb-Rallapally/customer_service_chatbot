"""FastAPI application factory for customer-support chat."""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from src.app import (
    ApplicationConfigurationError,
    ApplicationInitializationError,
    ApplicationServices,
    create_application,
)
from src.conversation import ConversationOwnershipError
from src.models import ChatResponse

from .schemas import ChatRequest
from .identity import (
    AuthenticatedIdentity,
    AuthenticationRequiredError,
    IdentityProvider,
    identity_provider_from_settings,
)
from .service import ChatApplicationService, ConversationServiceUnavailableError
from src.config import get_settings

logger = logging.getLogger(__name__)


def create_app(
    chat_service: ChatApplicationService,
    *,
    identity_provider: Optional[IdentityProvider] = None,
) -> Any:
    """Create the HTTP app with injected conversation orchestration."""

    app = FastAPI(title="Customer Service Chatbot API", version="0.1.0")
    provider = identity_provider or identity_provider_from_settings(get_settings())

    def get_authenticated_identity(request: Request) -> AuthenticatedIdentity:
        try:
            return provider.get_identity(request)
        except AuthenticationRequiredError as exc:
            raise HTTPException(status_code=401, detail="Authentication is required.") from exc
        except Exception as exc:
            logger.exception("Identity provider failed")
            raise HTTPException(
                status_code=503,
                detail="The support service is temporarily unavailable. Please try again.",
            ) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return process health without contacting external dependencies."""

        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def post_chat(
        request: ChatRequest,
        response: Response,
        identity: AuthenticatedIdentity = Depends(get_authenticated_identity),
    ) -> ChatResponse:
        try:
            if request.user_id is not None and request.user_id != identity.user_id:
                raise ConversationOwnershipError("request identity does not match")
            effective_request = request.model_copy(update={"user_id": identity.user_id})
            chat_response = chat_service.chat(effective_request)
            resolution_status = chat_service.resolution_status(request.conversation_id)
            if resolution_status is not None:
                response.headers["X-Resolution-Status"] = resolution_status
            return chat_response
        except (
            ConversationServiceUnavailableError,
            ApplicationConfigurationError,
            ApplicationInitializationError,
        ) as exc:
            logger.warning("Conversation service unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="The support service is temporarily unavailable. Please try again.",
            ) from exc
        except ConversationOwnershipError as exc:
            logger.warning("Conversation ownership check failed")
            raise HTTPException(
                status_code=403,
                detail="This conversation is not available to this user.",
            ) from exc
        except ValueError as exc:
            # Other ConversationEngine validation errors are client errors.
            logger.warning("Invalid conversation request")
            raise HTTPException(
                status_code=400, detail="Invalid conversation request."
            ) from exc
        except Exception:
            logger.exception("Unexpected failure while processing a chat request")
            raise HTTPException(
                status_code=500,
                detail="Unable to process the support request right now.",
            )

    return app


class _LazyChatApplicationService:
    """Create the composition graph on the first chat request, not on import."""

    def __init__(
        self, application_factory: Callable[[], ApplicationServices]
    ) -> None:
        self._application_factory = application_factory
        self._services: Optional[ApplicationServices] = None
        self._chat_service: Optional[ChatApplicationService] = None
        self._lock = RLock()

    def chat(self, request: ChatRequest) -> ChatResponse:
        with self._lock:
            if self._chat_service is None:
                self._services = self._application_factory()
                self._chat_service = ChatApplicationService(
                    self._services.conversation_engine
                )
            chat_service = self._chat_service
        return chat_service.chat(request)

    def resolution_status(self, conversation_id: str) -> Optional[str]:
        with self._lock:
            chat_service = self._chat_service
        return (
            chat_service.resolution_status(conversation_id)
            if chat_service is not None
            else None
        )

    def close(self) -> None:
        with self._lock:
            services = self._services
            self._services = None
            self._chat_service = None
        if services is not None:
            services.close()


def create_runtime_app(
    *,
    application_factory: Callable[[], ApplicationServices] = create_application,
    identity_provider: Optional[IdentityProvider] = None,
) -> Any:
    """Create the runnable API without initializing OCI or Oracle on import."""

    lazy_service = _LazyChatApplicationService(application_factory)
    app = create_app(lazy_service, identity_provider=identity_provider)
    app.add_event_handler("shutdown", lazy_service.close)
    return app


# Uvicorn entry point. This only creates FastAPI routes; the composition root
# is invoked lazily by the first /chat request.
app = create_runtime_app()
