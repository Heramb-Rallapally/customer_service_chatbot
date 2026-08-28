"""Errors raised by concrete LLM adapters."""


class LLMServiceError(RuntimeError):
    """Raised when an LLM provider cannot safely produce a usable response."""
