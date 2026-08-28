"""Oracle Database-backed, optimistic-concurrency conversation memory."""

from __future__ import annotations

import json
import re
import weakref
from threading import RLock
from typing import Any, Optional

from src.models import ConversationState

from .interfaces import ConversationSnapshot
from .memory_exceptions import (
    ConversationConflictError,
    ConversationNotFoundError,
    ConversationPersistenceError,
)


_ORACLE_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_$#]{0,127}")


class OracleConversationMemory:
    """Persist conversation state in a provisioned Oracle table.

    The table is deliberately provisioned through the documented SQL schema,
    not created as an application-import side effect.  A caller must use
    ``load_with_version`` and ``save_with_version`` for conflict-safe updates.
    """

    def __init__(self, connection: Any, *, table_name: str) -> None:
        if connection is None:
            raise ValueError("connection is required")
        if not _ORACLE_IDENTIFIER.fullmatch(table_name):
            raise ValueError("table_name must be a single valid Oracle identifier")
        self._connection = connection
        self._table_name = table_name
        self._loaded_versions: dict[int, tuple[weakref.ReferenceType[ConversationState], int]] = {}
        self._loaded_versions_lock = RLock()

    def load(self, conversation_id: str) -> Optional[ConversationState]:
        snapshot = self.load_with_version(conversation_id)
        if snapshot is None:
            return None
        state = snapshot.state.model_copy(deep=True)
        self._remember_loaded_version(state, snapshot.version)
        return state

    def load_with_version(
        self, conversation_id: str
    ) -> Optional[ConversationSnapshot]:
        """Load one conversation and its optimistic-concurrency version."""

        row = self._fetch_one(
            f"SELECT state_json, version FROM {self._table_name} "
            "WHERE conversation_id = :conversation_id",
            {"conversation_id": conversation_id},
        )
        if row is None:
            return None
        try:
            state_json, version = row[0], int(row[1])
            state = ConversationState.model_validate_json(self._clob_to_text(state_json))
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConversationPersistenceError("Stored conversation state is invalid") from exc
        if version < 1:
            raise ConversationPersistenceError("Stored conversation version is invalid")
        return ConversationSnapshot(state=state, version=version)

    def save(self, state: ConversationState) -> None:
        """Preserve legacy saves without silently allowing stale updates."""

        expected_version = self._take_loaded_version(state)
        if expected_version is not None:
            self.save_with_version(state, expected_version=expected_version)
            return
        if self.load_with_version(state.conversation_id) is None:
            self.save_with_version(state, expected_version=0)
            return
        raise ConversationPersistenceError(
            "Existing durable conversations must be loaded before they are saved"
        )

    def save_with_version(self, state: ConversationState, *, expected_version: int) -> int:
        """Insert or update state only when no concurrent writer has won."""

        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        state_json = self._serialize_state(state)
        if expected_version == 0:
            self._insert(state, state_json)
            return 1

        rowcount = self._execute(
            f"UPDATE {self._table_name} "
            "SET user_id = :user_id, state_json = :state_json, "
            "version = version + 1, updated_at = SYSTIMESTAMP "
            "WHERE conversation_id = :conversation_id AND version = :expected_version",
            {
                "conversation_id": state.conversation_id,
                "user_id": state.user_id,
                "state_json": state_json,
                "expected_version": expected_version,
            },
            commit=True,
        )
        if rowcount == 1:
            return expected_version + 1
        if self._conversation_exists(state.conversation_id):
            raise ConversationConflictError("Conversation was updated by another request")
        raise ConversationNotFoundError("Conversation does not exist")

    def get_summary(self, conversation_id: str) -> Optional[str]:
        row = self._fetch_one(
            f"SELECT summary FROM {self._table_name} WHERE conversation_id = :conversation_id",
            {"conversation_id": conversation_id},
        )
        if row is None or row[0] is None:
            return None
        try:
            return self._clob_to_text(row[0])
        except (TypeError, ValueError) as exc:
            raise ConversationPersistenceError("Stored conversation summary is invalid") from exc

    def list_for_user(
        self,
        user_id: str,
        *,
        exclude_conversation_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[ConversationState]:
        """Load a bounded set of states owned by one authenticated user only."""

        if limit < 1:
            return []
        statement = (
            "SELECT state_json FROM ("
            f"SELECT state_json FROM {self._table_name} WHERE user_id = :user_id "
            "ORDER BY updated_at DESC) WHERE ROWNUM <= :limit"
        )
        rows = self._fetch_all(statement, {"user_id": user_id, "limit": limit})
        states: list[ConversationState] = []
        try:
            for row in rows:
                state = ConversationState.model_validate_json(self._clob_to_text(row[0]))
                if (
                    state.user_id == user_id
                    and state.conversation_id != exclude_conversation_id
                ):
                    states.append(state)
        except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConversationPersistenceError("Stored conversation state is invalid") from exc
        return states[:limit]

    def set_summary(self, conversation_id: str, summary: Optional[str]) -> None:
        """Persist a future bounded summary and invalidate stale state writers."""

        rowcount = self._execute(
            f"UPDATE {self._table_name} "
            "SET summary = :summary, version = version + 1, updated_at = SYSTIMESTAMP "
            "WHERE conversation_id = :conversation_id",
            {"conversation_id": conversation_id, "summary": summary},
            commit=True,
        )
        if rowcount != 1:
            raise ConversationNotFoundError("Conversation does not exist")

    def _insert(self, state: ConversationState, state_json: str) -> None:
        try:
            self._execute(
                f"INSERT INTO {self._table_name} "
                "(conversation_id, user_id, state_json, version, created_at, updated_at) "
                "VALUES (:conversation_id, :user_id, :state_json, 1, SYSTIMESTAMP, SYSTIMESTAMP)",
                {
                    "conversation_id": state.conversation_id,
                    "user_id": state.user_id,
                    "state_json": state_json,
                },
                commit=True,
            )
        except ConversationPersistenceError as exc:
            if self._conversation_exists(state.conversation_id):
                raise ConversationConflictError(
                    "Conversation was created by another request"
                ) from exc
            raise

    def _conversation_exists(self, conversation_id: str) -> bool:
        return (
            self._fetch_one(
                f"SELECT 1 FROM {self._table_name} WHERE conversation_id = :conversation_id",
                {"conversation_id": conversation_id},
            )
            is not None
        )

    def _remember_loaded_version(self, state: ConversationState, version: int) -> None:
        state_id = id(state)

        def remove(reference: weakref.ReferenceType[ConversationState]) -> None:
            with self._loaded_versions_lock:
                current = self._loaded_versions.get(state_id)
                if current is not None and current[0] is reference:
                    self._loaded_versions.pop(state_id, None)

        reference = weakref.ref(state, remove)
        with self._loaded_versions_lock:
            self._loaded_versions[state_id] = (reference, version)

    def _take_loaded_version(self, state: ConversationState) -> Optional[int]:
        with self._loaded_versions_lock:
            remembered = self._loaded_versions.pop(id(state), None)
        if remembered is None or remembered[0]() is not state:
            return None
        return remembered[1]

    @staticmethod
    def _serialize_state(state: ConversationState) -> str:
        try:
            return json.dumps(
                state.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ConversationPersistenceError("Conversation state is not JSON serializable") from exc

    @staticmethod
    def _clob_to_text(value: Any) -> str:
        read = getattr(value, "read", None)
        text = read() if callable(read) else value
        if not isinstance(text, str):
            raise TypeError("stored value must be text")
        return text

    def _fetch_one(self, statement: str, parameters: dict[str, Any]) -> Any:
        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(statement, parameters)
            return cursor.fetchone()
        except ConversationPersistenceError:
            raise
        except Exception as exc:
            raise ConversationPersistenceError("Conversation persistence operation failed") from exc
        finally:
            self._close_cursor(cursor)

    def _fetch_all(self, statement: str, parameters: dict[str, Any]) -> list[Any]:
        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(statement, parameters)
            return list(cursor.fetchall())
        except Exception as exc:
            raise ConversationPersistenceError("Conversation persistence operation failed") from exc
        finally:
            self._close_cursor(cursor)

    def _execute(
        self, statement: str, parameters: dict[str, Any], *, commit: bool
    ) -> int:
        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(statement, parameters)
            rowcount = cursor.rowcount
            if commit:
                self._connection.commit()
            return int(rowcount) if rowcount is not None else 0
        except ConversationPersistenceError:
            raise
        except Exception as exc:
            self._rollback_quietly()
            raise ConversationPersistenceError("Conversation persistence operation failed") from exc
        finally:
            self._close_cursor(cursor)

    @staticmethod
    def _close_cursor(cursor: Any) -> None:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()

    def _rollback_quietly(self) -> None:
        rollback = getattr(self._connection, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass
