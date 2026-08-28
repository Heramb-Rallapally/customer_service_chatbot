"""Credential-free tests for the Oracle conversation-memory adapter."""

from __future__ import annotations

import json

import pytest

from src.conversation import (
    ConversationConflictError,
    ConversationNotFoundError,
    ConversationPersistenceError,
    OracleConversationMemory,
)
from src.models import (
    ConversationMessage,
    ConversationState,
    MessageRole,
    ResolutionStatus,
    Severity,
)


class FakeCursor:
    def __init__(self, connection: "FakeOracleConnection") -> None:
        self._connection = connection
        self._row: tuple[object, ...] | None = None
        self.rowcount: int | None = 0
        self.closed = False

    def execute(self, statement: str, parameters: dict[str, object]) -> None:
        if self._connection.failure is not None:
            raise self._connection.failure
        normalized = " ".join(statement.upper().split())
        conversation_id = str(parameters["conversation_id"]) if "conversation_id" in parameters else ""
        row = self._connection.rows.get(conversation_id)
        if normalized.startswith("SELECT STATE_JSON, VERSION"):
            self._row = None if row is None else (row["state_json"], row["version"])
        elif normalized.startswith("SELECT SUMMARY"):
            self._row = None if row is None else (row["summary"],)
        elif normalized.startswith("SELECT 1"):
            self._row = None if row is None else (1,)
        elif normalized.startswith("SELECT STATE_JSON FROM ("):
            user_id = parameters["user_id"]
            limit = int(parameters["limit"])
            self._rows = [
                (stored["state_json"],)
                for stored in self._connection.rows.values()
                if stored["user_id"] == user_id
            ][:limit]
        elif normalized.startswith("INSERT INTO"):
            if row is not None:
                raise RuntimeError("ORA-00001 duplicate key")
            self._connection.rows[conversation_id] = {
                "user_id": parameters["user_id"],
                "state_json": parameters["state_json"],
                "summary": None,
                "version": 1,
            }
            self.rowcount = 1
        elif normalized.startswith("UPDATE") and "SET SUMMARY" in normalized:
            if row is None:
                self.rowcount = 0
            else:
                row["summary"] = parameters["summary"]
                row["version"] = int(row["version"]) + 1
                self.rowcount = 1
        elif normalized.startswith("UPDATE"):
            if row is None or row["version"] != parameters["expected_version"]:
                self.rowcount = 0
            else:
                row["user_id"] = parameters["user_id"]
                row["state_json"] = parameters["state_json"]
                row["version"] = int(row["version"]) + 1
                self.rowcount = 1
        else:  # pragma: no cover - protects fake coverage as SQL evolves.
            raise AssertionError(statement)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._row

    def fetchall(self) -> list[tuple[object, ...]]:
        return getattr(self, "_rows", [])

    def close(self) -> None:
        self.closed = True


class FakeOracleConnection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.commit_calls = 0
        self.rollback_calls = 0
        self.failure: Exception | None = None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


def rich_state(conversation_id: str = "conversation-1") -> ConversationState:
    return ConversationState(
        conversation_id=conversation_id,
        user_id="user-a",
        messages=[
            ConversationMessage(role=MessageRole.USER, content="VPN fails."),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Check the token."),
        ],
        product="Oracle VPN",
        version="5.2",
        issue_type="authentication",
        issue_summary="VPN authentication failure",
        severity=Severity.HIGH,
        resolution_status=ResolutionStatus.AWAITING_CONFIRMATION,
        troubleshooting_steps=["Check the token"],
        attempted_steps=["Restart the client"],
        turn_count=4,
    )


def test_oracle_memory_round_trips_all_conversation_state_fields() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")
    state = rich_state()

    assert memory.save_with_version(state, expected_version=0) == 1
    snapshot = memory.load_with_version(state.conversation_id)

    assert snapshot is not None
    assert snapshot.version == 1
    assert snapshot.state == state
    persisted = json.loads(str(connection.rows[state.conversation_id]["state_json"]))
    assert persisted["severity"] == "HIGH"
    assert persisted["messages"][0]["role"] == "USER"
    assert connection.commit_calls == 1


def test_oracle_memory_accepts_native_json_objects_returned_by_oracle_23ai() -> None:
    connection = FakeOracleConnection()
    state = rich_state()
    connection.rows[state.conversation_id] = {
        "user_id": state.user_id,
        "state_json": state.model_dump(mode="json"),
        "summary": None,
        "version": 1,
    }

    snapshot = OracleConversationMemory(
        connection, table_name="CHAT_CONVERSATIONS"
    ).load_with_version(state.conversation_id)

    assert snapshot is not None
    assert snapshot.state == state
    assert snapshot.version == 1


def test_oracle_memory_updates_existing_state_and_persists_summary() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")
    state = rich_state()
    memory.save_with_version(state, expected_version=0)
    snapshot = memory.load_with_version(state.conversation_id)
    assert snapshot is not None

    snapshot.state.resolution_status = ResolutionStatus.RESOLVED
    assert memory.save_with_version(snapshot.state, expected_version=snapshot.version) == 2
    memory.set_summary(state.conversation_id, "Customer confirmed the resolution.")

    updated = memory.load_with_version(state.conversation_id)
    assert updated is not None
    assert updated.version == 3
    assert updated.state.resolution_status is ResolutionStatus.RESOLVED
    assert memory.get_summary(state.conversation_id) == "Customer confirmed the resolution."


def test_oracle_memory_lists_only_the_requested_users_states() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")
    first = rich_state("conversation-1")
    second = rich_state("conversation-2")
    other = rich_state("conversation-3")
    other.user_id = "user-b"
    for state in (first, second, other):
        memory.save_with_version(state, expected_version=0)

    history = memory.list_for_user("user-a", exclude_conversation_id="conversation-2")

    assert [state.conversation_id for state in history] == ["conversation-1"]
    assert memory.list_for_user("user-b")[0].conversation_id == "conversation-3"


def test_oracle_memory_handles_missing_malformed_and_non_json_state_safely() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")

    assert memory.load("missing") is None
    with pytest.raises(ConversationNotFoundError):
        memory.save_with_version(rich_state("missing"), expected_version=1)

    connection.rows["broken"] = {
        "user_id": "user-a",
        "state_json": "not-json",
        "summary": None,
        "version": 1,
    }
    with pytest.raises(ConversationPersistenceError, match="invalid"):
        memory.load("broken")

    class NonJsonState:
        conversation_id = "bad-state"
        user_id = None

        @staticmethod
        def model_dump(*, mode: str) -> dict[str, float]:
            assert mode == "json"
            return {"invalid": float("nan")}

    with pytest.raises(ConversationPersistenceError, match="JSON serializable"):
        memory.save_with_version(NonJsonState(), expected_version=0)  # type: ignore[arg-type]


def test_oracle_memory_detects_stale_writers_and_keeps_multiple_conversations() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")
    first_state = rich_state("conversation-1")
    second_state = rich_state("conversation-2")
    memory.save_with_version(first_state, expected_version=0)
    memory.save_with_version(second_state, expected_version=0)
    first_snapshot = memory.load_with_version("conversation-1")
    stale_snapshot = memory.load_with_version("conversation-1")
    assert first_snapshot is not None and stale_snapshot is not None

    first_snapshot.state.product = "Updated VPN"
    memory.save_with_version(first_snapshot.state, expected_version=first_snapshot.version)
    stale_snapshot.state.product = "Stale VPN"
    with pytest.raises(ConversationConflictError):
        memory.save_with_version(stale_snapshot.state, expected_version=stale_snapshot.version)

    second = memory.load("conversation-2")
    assert second is not None
    assert second.user_id == "user-a"
    assert second.product == "Oracle VPN"


def test_oracle_memory_hides_database_errors_and_preserves_safe_legacy_saves() -> None:
    connection = FakeOracleConnection()
    memory = OracleConversationMemory(connection, table_name="CHAT_CONVERSATIONS")
    connection.failure = RuntimeError("ORA-12154 db.example.invalid/password")

    with pytest.raises(ConversationPersistenceError) as error:
        memory.load("conversation-1")
    assert "db.example.invalid" not in str(error.value)
    assert connection.rollback_calls == 0

    connection.failure = None
    memory.save(rich_state())
    loaded = memory.load("conversation-1")
    assert loaded is not None
    loaded.product = "Updated VPN"
    memory.save(loaded)
    updated = memory.load("conversation-1")
    assert updated is not None
    assert updated.product == "Updated VPN"

    with pytest.raises(ConversationPersistenceError, match="must be loaded"):
        memory.save(rich_state())
