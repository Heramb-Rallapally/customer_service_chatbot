"""Concrete LLM adapters consumed through the conversation LLMService port."""

from .exceptions import LLMServiceError
from .oci_cohere import OciCohereLLMService

__all__ = ["LLMServiceError", "OciCohereLLMService"]
