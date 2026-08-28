"""Local Ollama adapter for the existing Conversation Engine LLM port."""

from __future__ import annotations

from src.config import Settings
from src.conversation.interfaces import GeneratedResponse, GenerationContext
from src.ollama import OllamaApiClient, OllamaClient, OllamaClientError

from .exceptions import LLMServiceError
from .grounding import build_grounded_prompt, parse_generated_response


class OllamaLLMService:
    """Generate grounded responses with a locally hosted Ollama model."""

    def __init__(
        self,
        client: OllamaClient,
        *,
        model_id: str = "llama3.2:3b",
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        self._client = client
        self._model_id = model_id

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaLLMService":
        model = settings.llm_model or "llama3.2:3b"
        client = OllamaApiClient(
            settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        try:
            client.ensure_models([model])
        except OllamaClientError as exc:
            client.close()
            raise LLMServiceError(
                "Ollama is unavailable or the LLM model is not installed"
            ) from exc
        return cls(client, model_id=model)

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        prompt = build_grounded_prompt(context)
        try:
            generated_text = self._client.chat(model=self._model_id, prompt=prompt)
        except OllamaClientError as exc:
            raise LLMServiceError("Ollama text generation request failed") from exc
        return parse_generated_response(
            generated_text,
            context.retrieved_knowledge,
            context.excluded_steps,
            provider_name="Ollama",
        )
