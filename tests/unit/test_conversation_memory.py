"""Unit tests for conversation state memory."""

from src.conversation.memory import InMemoryConversationMemory
from src.conversation.memory_exceptions import ConversationConflictError
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


def test_in_memory_versioned_operations_detect_stale_writers() -> None:
    memory = InMemoryConversationMemory()
    state = ConversationState(conversation_id="conversation-1", product="Oracle VPN")

    assert memory.save_with_version(state, expected_version=0) == 1
    first = memory.load_with_version("conversation-1")
    second = memory.load_with_version("conversation-1")
    assert first is not None and second is not None

    first.state.version = "5.2"
    assert memory.save_with_version(first.state, expected_version=first.version) == 2

    second.state.version = "5.3"
    try:
        memory.save_with_version(second.state, expected_version=second.version)
    except ConversationConflictError:
        pass
    else:  # pragma: no cover - the assertion gives a clearer failure message.
        raise AssertionError("a stale state must not overwrite a newer state")
