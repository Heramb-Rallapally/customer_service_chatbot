"""OCI Generative AI adapter for the Conversation Engine LLMService port."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.config import Settings
from src.conversation.interfaces import GeneratedResponse, GenerationContext

from .exceptions import LLMServiceError
from .grounding import build_grounded_prompt, parse_generated_response


class OciCohereLLMService:
    """Generate grounded Cohere responses from an existing ``GenerationContext``.

    The adapter deliberately returns only message text, suggested actions, and
    confidence. Citation construction remains the Conversation Engine's
    responsibility because it owns the retrieved ``RetrievalResult`` values.
    """

    def __init__(
        self,
        client: Any,
        *,
        compartment_id: str,
        model_id: str,
        request_factory: Callable[[str], Any],
    ) -> None:
        if not compartment_id.strip() or not model_id.strip():
            raise ValueError("compartment_id and model_id must not be blank")
        self._client = client
        self._compartment_id = compartment_id
        self._model_id = model_id
        self._request_factory = request_factory

    @classmethod
    def from_settings(cls, settings: Settings) -> "OciCohereLLMService":
        """Create an OCI SDK-backed Cohere service from centralized settings.

        OCI imports and credential loading are deferred so importing this
        package never requires local OCI credentials.
        """

        if not settings.oci_compartment_id or not settings.llm_model:
            raise LLMServiceError("OCI_COMPARTMENT_ID and LLM_MODEL must be configured")
        try:
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import (
                CohereLlmInferenceRequest,
                GenerateTextDetails,
                OnDemandServingMode,
            )
        except ImportError as exc:
            raise LLMServiceError("Install the optional 'oci' package to use OCI generation") from exc

        config = oci.config.from_file(profile_name=settings.oci_config_profile)
        client = GenerativeAiInferenceClient(config, service_endpoint=settings.oci_endpoint)

        def request_factory(prompt: str) -> Any:
            return GenerateTextDetails(
                compartment_id=settings.oci_compartment_id,
                serving_mode=OnDemandServingMode(model_id=settings.llm_model),
                inference_request=CohereLlmInferenceRequest(
                    prompt=prompt,
                    is_stream=False,
                    num_generations=1,
                    max_tokens=700,
                    temperature=0.2,
                    return_likelihoods="NONE",
                ),
            )

        return cls(
            client,
            compartment_id=settings.oci_compartment_id,
            model_id=settings.llm_model,
            request_factory=request_factory,
        )

    def generate(self, context: GenerationContext) -> GeneratedResponse:
        """Request one structured, grounded response from OCI Generative AI."""

        prompt = build_grounded_prompt(context)
        try:
            response = self._client.generate_text(self._request_factory(prompt))
            generated_text = _response_text(response)
        except LLMServiceError:
            raise
        except Exception as exc:  # OCI transport and service errors vary by SDK version.
            raise LLMServiceError("OCI text generation request failed") from exc
        return parse_generated_response(
            generated_text,
            context.retrieved_knowledge,
            context.excluded_steps,
            provider_name="OCI",
        )


def _response_text(response: Any) -> str:
    try:
        generated_texts = response.data.inference_response.generated_texts
        text = generated_texts[0].text
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMServiceError("OCI returned an unexpected text-generation response") from exc
    if not isinstance(text, str) or not text.strip():
        raise LLMServiceError("OCI returned an empty text-generation response")
    return text.strip()
