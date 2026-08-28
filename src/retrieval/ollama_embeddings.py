"""Ollama embedding adapter for the existing retrieval port."""

from __future__ import annotations

import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.ollama import OllamaApiClient, OllamaClient, OllamaClientError

from .exceptions import EmbeddingError


class OllamaEmbeddingService(Embeddings):
    """Create validated embeddings through Ollama's local ``/api/embed`` API."""

    def __init__(
        self,
        client: OllamaClient,
        *,
        model_id: str = "nomic-embed-text",
        embedding_dimension: int = 768,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self._client = client
        self._model_id = model_id
        self.embedding_dimension = embedding_dimension

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaEmbeddingService":
        model = settings.embedding_model or "nomic-embed-text"
        dimension = settings.embedding_dimension or 768
        client = OllamaApiClient(
            settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        try:
            client.ensure_models([model])
        except OllamaClientError as exc:
            client.close()
            raise EmbeddingError(
                "Ollama is unavailable or the embedding model is not installed"
            ) from exc
        return cls(client, model_id=model, embedding_dimension=dimension)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts must contain at least one non-empty value")
        try:
            vectors = self._client.embed(model=self._model_id, texts=texts)
        except OllamaClientError as exc:
            raise EmbeddingError("Ollama embedding request failed") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of embeddings")
        normalized: list[list[float]] = []
        for raw_vector in vectors:
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingError("Ollama returned an empty embedding")
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Ollama returned a non-numeric embedding value") from exc
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Ollama returned a non-finite embedding value")
            if len(vector) != self.embedding_dimension:
                raise EmbeddingError(
                    f"Ollama embedding dimension {len(vector)} does not match expected "
                    f"dimension {self.embedding_dimension}"
                )
            normalized.append(vector)
        return normalized
