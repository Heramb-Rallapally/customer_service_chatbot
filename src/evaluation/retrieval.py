"""Transparent retrieval observation for end-to-end evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Protocol, Sequence

from src.conversation.interfaces import Retriever
from src.models import KnowledgeDocument, RetrievalResult


@dataclass(frozen=True)
class RetrievalObservation:
    query: str
    filters: dict[str, str]
    top_k: int
    results: tuple[RetrievalResult, ...]


class RetrievalObservationSource(Protocol):
    def checkpoint(self) -> int:
        """Return a stable cursor before an evaluation case begins."""

    def observations_since(self, checkpoint: int) -> Sequence[RetrievalObservation]:
        """Return copies of observations recorded after the cursor."""


class RecordingRetriever:
    """Delegate each real search exactly once while retaining top-k result copies."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._observations: list[RetrievalObservation] = []
        self._lock = RLock()

    def search(
        self, *, query: str, filters: Mapping[str, str], top_k: int
    ) -> Sequence[RetrievalResult]:
        results = tuple(
            result.model_copy(deep=True)
            for result in self._retriever.search(
                query=query, filters=filters, top_k=top_k
            )
        )
        observation = RetrievalObservation(
            query=query,
            filters=dict(filters),
            top_k=top_k,
            results=results,
        )
        with self._lock:
            self._observations.append(observation)
        return tuple(result.model_copy(deep=True) for result in results)

    def index_documents(self, documents: Sequence[KnowledgeDocument]) -> None:
        """Preserve the wrapped indexer contract for composition-root use."""

        indexer = getattr(self._retriever, "index_documents", None)
        if not callable(indexer):
            raise TypeError("wrapped retriever does not support document indexing")
        indexer(documents)

    def checkpoint(self) -> int:
        with self._lock:
            return len(self._observations)

    def observations_since(self, checkpoint: int) -> Sequence[RetrievalObservation]:
        if checkpoint < 0:
            raise ValueError("checkpoint must not be negative")
        with self._lock:
            return tuple(
                RetrievalObservation(
                    query=item.query,
                    filters=dict(item.filters),
                    top_k=item.top_k,
                    results=tuple(result.model_copy(deep=True) for result in item.results),
                )
                for item in self._observations[checkpoint:]
            )
