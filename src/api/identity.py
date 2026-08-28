"""Injectable authenticated-identity boundary for the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastapi import Request

from src.config import Settings


class AuthenticationRequiredError(RuntimeError):
    """Raised when no trusted identity is available for a chat request."""


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """A validated identity established outside the chat request payload."""

    user_id: str

    def __post_init__(self) -> None:
        normalized_user_id = self.user_id.strip()
        if not normalized_user_id:
            raise ValueError("authenticated user_id must not be blank")
        object.__setattr__(self, "user_id", normalized_user_id)


class IdentityProvider(Protocol):
    """Resolve a trusted identity from server-controlled request context."""

    def get_identity(self, request: Request) -> AuthenticatedIdentity:
        """Return the caller identity or raise ``AuthenticationRequiredError``."""


class DevelopmentIdentityProvider:
    """Use an explicit server-side development identity, never request JSON."""

    def __init__(self, user_id: str) -> None:
        self._identity = AuthenticatedIdentity(user_id=user_id)

    def get_identity(self, request: Request) -> AuthenticatedIdentity:
        return self._identity


class RequestStateIdentityProvider:
    """Read identity populated by trusted production authentication middleware."""

    def get_identity(self, request: Request) -> AuthenticatedIdentity:
        identity = getattr(request.state, "authenticated_identity", None)
        if isinstance(identity, AuthenticatedIdentity):
            return identity
        raise AuthenticationRequiredError("No authenticated identity is available")


def identity_provider_from_settings(settings: Settings) -> IdentityProvider:
    """Select local development identity or production middleware integration."""

    mode = settings.api_auth_mode.strip().lower()
    if mode == "development":
        return DevelopmentIdentityProvider(settings.api_development_user_id)
    if mode == "required":
        return RequestStateIdentityProvider()
    raise ValueError("API_AUTH_MODE must be 'development' or 'required'")
