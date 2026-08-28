"""Errors raised by the knowledge ingestion pipeline."""


class IngestionError(ValueError):
    """Base error for invalid ingestion input."""


class IndexingError(IngestionError):
    """Raised when an indexing request does not contain shared knowledge documents."""


class UnsupportedSourceError(IngestionError):
    """Raised when a source format cannot be loaded locally."""
