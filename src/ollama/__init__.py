"""Dependency-light local Ollama API client."""

from .client import (
    OllamaApiClient,
    OllamaClient,
    OllamaClientError,
    OllamaModelUnavailableError,
)

__all__ = [
    "OllamaApiClient",
    "OllamaClient",
    "OllamaClientError",
    "OllamaModelUnavailableError",
]
