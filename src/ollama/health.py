"""Command-line availability check for configured Ollama models."""

from __future__ import annotations

import sys

from src.config import get_settings

from .client import OllamaApiClient, OllamaClientError


def main() -> int:
    settings = get_settings()
    models: list[str] = []
    if settings.embedding_provider.strip().lower() == "ollama":
        models.append(settings.embedding_model or "nomic-embed-text")
    if settings.llm_provider.strip().lower() == "ollama":
        models.append(settings.llm_model or "llama3.2:3b")
    if not models:
        print("Ollama is not selected by the current provider configuration.")
        return 0

    try:
        client = OllamaApiClient(
            settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    except ValueError as exc:
        print(f"Ollama health check failed: {exc}", file=sys.stderr)
        return 1
    try:
        client.ensure_models(models)
    except OllamaClientError as exc:
        print(f"Ollama health check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    print("Ollama health check passed for: " + ", ".join(models))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command.
    raise SystemExit(main())
