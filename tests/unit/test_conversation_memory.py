"""Unit tests for conversation state memory."""

from src.conversation.memory import InMemoryConversationMemory
from src.models import ConversationState


def test_memory_returns_independent_state_copies() -> None:
    memory = InMemoryConversationMemory()
    original = ConversationState(conversation_id="conversation-1", product="Oracle VPN")
    memory.save(original)

    loaded = memory.load("conversation-1")
    assert loaded is not None
    loaded.product = "Changed"

    reloaded = memory.load("conversation-1")
    assert reloaded is not None
    assert reloaded.product == "Oracle VPN"


def test_memory_supports_optional_future_summary() -> None:
    memory = InMemoryConversationMemory()

    assert memory.get_summary("conversation-1") is None
    memory.set_summary("conversation-1", "Customer uses Oracle VPN.")

    assert memory.get_summary("conversation-1") == "Customer uses Oracle VPN."

