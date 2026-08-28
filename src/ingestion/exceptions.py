"""Errors raised by the knowledge ingestion pipeline."""


class IngestionError(ValueError):
    """Base error for invalid ingestion input."""


class UnsupportedSourceError(IngestionError):
    """Raised when a source format cannot be loaded locally."""
