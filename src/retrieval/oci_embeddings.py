"""OCI Generative AI embedding adapter, imported only when configured for use."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config import Settings

from .exceptions import EmbeddingError


class OCIEmbeddingService(Embeddings):
    """Embeds text with OCI Generative AI while keeping OCI optional for tests."""

    def __init__(
        self, client: Any, *, compartment_id: str, model_id: str,
        request_factory: Callable[[Sequence[str]], Any],
        embedding_dimension: int | None = None,
    ) -> None:
        self._client = client
        self._compartment_id = compartment_id
        self._model_id = model_id
        self._request_factory = request_factory
        if embedding_dimension is not None and embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self.embedding_dimension = embedding_dimension

    @classmethod
    def from_settings(cls, settings: Settings) -> "OCIEmbeddingService":
        """Create an OCI SDK-backed embedder from centralized settings.

        The OCI SDK is imported here instead of at module import time, allowing
        mocked development and unit tests without OCI installed or configured.
        """

        if not settings.oci_compartment_id or not settings.embedding_model:
            raise EmbeddingError("OCI_COMPARTMENT_ID and EMBEDDING_MODEL must be configured")
        try:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode
        except ImportError as exc:
            raise EmbeddingError("Install the optional 'oci' package to use OCI embeddings") from exc

        config = oci.config.from_file(profile_name=settings.oci_config_profile)
        client = GenerativeAiInferenceClient(config, service_endpoint=settings.oci_endpoint)

        def request_factory(texts: Sequence[str]) -> Any:
            return EmbedTextDetails(
                compartment_id=settings.oci_compartment_id,
                serving_mode=OnDemandServingMode(model_id=settings.embedding_model),
                inputs=list(texts),
            )

        return cls(
            client,
            compartment_id=settings.oci_compartment_id,
            model_id=settings.embedding_model,
            request_factory=request_factory,
            embedding_dimension=settings.embedding_dimension,
        )

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
            response = self._client.embed_text(self._request_factory(texts))
            vectors = response.data.embeddings
        except Exception as exc:  # OCI errors vary by SDK version and transport.
            raise EmbeddingError("OCI embedding request failed") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError("OCI returned an unexpected number of embeddings")
        normalised = [list(vector) for vector in vectors]
        for vector in normalised:
            if not vector:
                raise EmbeddingError("OCI returned an empty embedding")
            try:
                values_are_finite = all(math.isfinite(float(value)) for value in vector)
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("OCI returned a non-numeric embedding value") from exc
            if not values_are_finite:
                raise EmbeddingError("OCI returned a non-finite embedding value")
            if self.embedding_dimension is None:
                self.embedding_dimension = len(vector)
            elif len(vector) != self.embedding_dimension:
                raise EmbeddingError(
                    "OCI embedding dimension does not match configured/previous dimension "
                    f"{self.embedding_dimension}"
                )
        return normalised
