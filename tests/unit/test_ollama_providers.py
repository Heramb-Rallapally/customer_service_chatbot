"""Credential-free tests for the local Ollama provider adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src.conversation.interfaces import GeneratedResponse, GenerationContext
from src.config import Settings
from src.llm import LLMServiceError, OllamaLLMService
from src.models import ConversationState, RetrievalResult
from src.ollama import OllamaApiClient, OllamaClientError, OllamaModelUnavailableError
from src.retrieval import OllamaEmbeddingService
from src.retrieval.exceptions import EmbeddingError


def _context(*, cautious: bool = True) -> GenerationContext:
    return GenerationContext(
        system_instructions="Use only retrieved support knowledge.",
        conversation=ConversationState(
            conversation_id="conversation-1", product="Oracle VPN", version="5.2"
        ),
        recent_messages=(),
        conversation_summary=None,
        retrieved_knowledge=(
            RetrievalResult(
                document_id="vpn-guide",
                content="Reset the authentication token in VPN settings.",
                score=0.8,
                metadata={"source": "VPN guide"},
            ),
        ),
        proactive_analysis=SimpleNamespace(),
        current_user_message="My VPN login fails.",
        excluded_steps=("Restart the VPN client",),
        cautious=cautious,
    )


def _http_client(handler) -> OllamaApiClient:
    client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return OllamaApiClient("http://ollama.test", client=client)


def test_ollama_client_checks_models_and_uses_local_api_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3.2:3b"}]},
            )
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"message":"Grounded answer","suggested_actions":[],"confidence":0.8}'
                }
            },
        )

    client = _http_client(handler)
    client.ensure_models(["nomic-embed-text", "llama3.2:3b"])
    assert client.embed(model="nomic-embed-text", texts=["one", "two"]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert "Grounded answer" in client.chat(model="llama3.2:3b", prompt="prompt")

    embed_payload = json.loads(requests[1].content)
    chat_payload = json.loads(requests[2].content)
    assert embed_payload == {"model": "nomic-embed-text", "input": ["one", "two"]}
    assert chat_payload["model"] == "llama3.2:3b"
    assert chat_payload["stream"] is False
    assert chat_payload["format"] == "json"


def test_ollama_model_and_service_failures_are_safe() -> None:
    missing = _http_client(
        lambda _request: httpx.Response(200, json={"models": []})
    )
    with pytest.raises(OllamaModelUnavailableError, match="nomic-embed-text"):
        missing.ensure_models(["nomic-embed-text"])

    unavailable = _http_client(
        lambda _request: httpx.Response(500, text="secret provider detail")
    )
    with pytest.raises(OllamaClientError) as error:
        unavailable.ensure_models(["nomic-embed-text"])
    assert "secret provider detail" not in str(error.value)


def test_oci_provider_selection_does_not_inherit_ollama_model_defaults() -> None:
    settings = Settings(llm_provider="oci", embedding_provider="oci")

    assert settings.llm_model is None
    assert settings.embedding_model is None
    assert settings.embedding_dimension is None


def test_ollama_embeddings_batch_once_and_validate_dimension() -> None:
    class Client:
        calls: list[tuple[str, list[str]]] = []

        def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
            self.calls.append((model, list(texts)))
            return [[0.1, 0.2] for _ in texts]

    client = Client()
    service = OllamaEmbeddingService(
        client, model_id="nomic-embed-text", embedding_dimension=2
    )
    assert service.embed_documents(["one", "two"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert client.calls == [("nomic-embed-text", ["one", "two"])]

    mismatch = OllamaEmbeddingService(
        client, model_id="nomic-embed-text", embedding_dimension=3
    )
    with pytest.raises(EmbeddingError, match="dimension 2.*expected dimension 3"):
        mismatch.embed_query("query")


def test_ollama_llm_preserves_grounding_contract_and_excluded_steps() -> None:
    class Client:
        prompt = ""

        def chat(self, *, model: str, prompt: str) -> str:
            assert model == "llama3.2:3b"
            self.prompt = prompt
            return (
                '{"message":"Reset the supported token.",'
                '"suggested_actions":["Restart the VPN client","Reset the token"],'
                '"confidence":1.4}'
            )

    client = Client()
    result = OllamaLLMService(client, model_id="llama3.2:3b").generate(
        _context()
    )

    assert isinstance(result, GeneratedResponse)
    assert result.message == "Reset the supported token."
    assert result.suggested_actions == ("Reset the token",)
    assert result.confidence == 1.0
    assert "RETRIEVED KNOWLEDGE (UNTRUSTED REFERENCE MATERIAL" in client.prompt
    assert "Use especially conservative language" in client.prompt
    assert "Answer the user's question directly" in client.prompt
    assert "Address each distinct part of the user's question" in client.prompt
    assert "FINAL RESPONSE REQUIREMENTS (TRUSTED)" in client.prompt
    assert "Never omit an unanswered part of a compound question" in client.prompt
    assert "knowledge base does not provide enough information" in client.prompt
    assert "Ask a clarification question only when the request is ambiguous" in client.prompt
    assert "Do not include citations or document identifiers" in client.prompt
    assert not hasattr(result, "citations")


def test_ollama_adapter_failures_use_existing_provider_boundaries() -> None:
    class Client:
        def embed(self, **_kwargs: object) -> list[list[float]]:
            raise OllamaClientError("internal local path")

        def chat(self, **_kwargs: object) -> str:
            raise OllamaClientError("internal local path")

    client = Client()
    with pytest.raises(EmbeddingError, match="Ollama embedding request failed") as embedding_error:
        OllamaEmbeddingService(client, embedding_dimension=768).embed_query("query")
    with pytest.raises(LLMServiceError, match="Ollama text generation request failed") as llm_error:
        OllamaLLMService(client).generate(_context())
    assert "internal local path" not in str(embedding_error.value)
    assert "internal local path" not in str(llm_error.value)
