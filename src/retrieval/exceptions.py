"""Retrieval-specific exceptions safe for API-layer handling."""


class RetrievalError(RuntimeError):
    """Base error for retrieval operations."""


class EmbeddingError(RetrievalError):
    """Raised when an embedding provider cannot create vectors."""


class VectorStoreError(RetrievalError):
    """Raised when a vector-store operation cannot complete."""
