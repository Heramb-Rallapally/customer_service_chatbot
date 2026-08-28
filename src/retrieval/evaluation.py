"""Repeatable, dependency-free retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from .filters import RetrievalFilters
from .service import RetrievalService


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """A query with one or more relevant knowledge-document IDs."""

    query: str
    relevant_document_ids: frozenset[str]
    filters: Optional[RetrievalFilters] = None


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Aggregate retrieval quality for a fixed dataset and cutoff."""

    recall_at_k: float
    mrr: float
    evaluated_queries: int


def evaluate_retrieval(
    service: RetrievalService, cases: Sequence[RetrievalEvaluationCase], *, k: int = 5
) -> RetrievalEvaluationReport:
    """Calculate Recall@K and mean reciprocal rank for supplied relevance labels."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not cases:
        return RetrievalEvaluationReport(recall_at_k=0.0, mrr=0.0, evaluated_queries=0)
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    for case in cases:
        ranked_ids = [result.document_id for result in service.retrieve(case.query, k=k, filters=case.filters)]
        relevant = case.relevant_document_ids
        recall_total += len(set(ranked_ids) & relevant) / len(relevant) if relevant else 0.0
        for rank, document_id in enumerate(ranked_ids, start=1):
            if document_id in relevant:
                reciprocal_rank_total += 1.0 / rank
                break
    return RetrievalEvaluationReport(
        recall_at_k=recall_total / len(cases), mrr=reciprocal_rank_total / len(cases), evaluated_queries=len(cases)
    )
