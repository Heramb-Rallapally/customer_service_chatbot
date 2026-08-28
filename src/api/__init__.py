"""HTTP boundary for the customer-support application."""

from .app import create_app, create_runtime_app
from .identity import (
    AuthenticatedIdentity,
    AuthenticationRequiredError,
    DevelopmentIdentityProvider,
    IdentityProvider,
    RequestStateIdentityProvider,
)
from .schemas import ChatRequest, FeedbackRequest, FeedbackResponse
from .service import (
    ChatApplicationService,
    ConversationService,
    ConversationServiceUnavailableError,
)

__all__ = [
    "ChatApplicationService",
    "AuthenticatedIdentity",
    "AuthenticationRequiredError",
    "ChatRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "ConversationService",
    "ConversationServiceUnavailableError",
    "DevelopmentIdentityProvider",
    "IdentityProvider",
    "RequestStateIdentityProvider",
    "create_app",
    "create_runtime_app",
]
