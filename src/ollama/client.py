"""Small typed wrapper around Ollama's local HTTP API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Protocol

import httpx


class OllamaClientError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


class OllamaModelUnavailableError(OllamaClientError):
    """Raised when a configured model is not installed locally."""


class OllamaClient(Protocol):
    """Provider port shared by the local LLM and embedding adapters."""

    def ensure_models(self, models: Sequence[str]) -> None: ...

    def embed(self, *, model: str, texts: Sequence[str]) -> list[list[float]]: ...

    def chat(self, *, model: str, prompt: str) -> str: ...

    def close(self) -> None: ...


class OllamaApiClient:
    """Access Ollama without requiring a provider-specific Python SDK."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Ollama base URL must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    def ensure_models(self, models: Sequence[str]) -> None:
        """Fail early with a safe, actionable error for missing local models."""

        payload = self._request("GET", "/api/tags")
        installed = {
            str(item.get("name") or item.get("model"))
            for item in payload.get("models", [])
            if isinstance(item, dict) and (item.get("name") or item.get("model"))
        }
        missing = [model for model in models if not _model_is_available(model, installed)]
        if missing:
            raise OllamaModelUnavailableError(
                "Required Ollama model is not installed: " + ", ".join(missing)
            )

    def embed(self, *, model: str, texts: Sequence[str]) -> list[list[float]]:
        payload = self._request(
            "POST", "/api/embed", json={"model": model, "input": list(texts)}
        )
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list):
            raise OllamaClientError("Ollama returned an invalid embedding response")
        return vectors

    def chat(self, *, model: str, prompt: str) -> str:
        payload = self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 700},
            },
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaClientError("Ollama returned an empty chat response")
        return content.strip()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaClientError(
                "Ollama is unavailable or returned an invalid response"
            ) from exc
        if not isinstance(payload, dict):
            raise OllamaClientError("Ollama returned an invalid response")
        if payload.get("error"):
            raise OllamaClientError("Ollama could not complete the request")
        return payload


def _model_is_available(requested: str, installed: set[str]) -> bool:
    return requested in installed or (
        ":" not in requested and f"{requested}:latest" in installed
    )
