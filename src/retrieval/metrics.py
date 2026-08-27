"""Similarity metrics supported by retrieval vector-store adapters."""

from __future__ import annotations

import math
from enum import Enum


class SimilarityMetric(str, Enum):
    """Vector-comparison strategies accepted by the retrieval layer."""

    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"
    DOT = "DOT"


class OracleScoreSemantics(str, Enum):
    """Meaning of raw scores returned by an OracleVS backend."""

    DISTANCE = "DISTANCE"
    SIMILARITY = "SIMILARITY"


def normalized_similarity(raw_score: float, metric: SimilarityMetric) -> float:
    """Convert a raw similarity score to a bounded, higher-is-better score.

    COSINE is expected to be in ``[-1, 1]``. DOT can be unbounded, so a
    monotonic sigmoid is used. EUCLIDEAN is already converted from distance by
    the caller before reaching this helper.
    """

    if metric is SimilarityMetric.COSINE:
        return _clamp((raw_score + 1.0) / 2.0)
    if metric is SimilarityMetric.DOT:
        return _sigmoid(raw_score)
    return _clamp(raw_score)


def normalized_distance(raw_distance: float, metric: SimilarityMetric) -> float:
    """Convert an Oracle vector distance to a `[0, 1]` similarity score."""

    if metric is SimilarityMetric.COSINE:
        # Oracle cosine distance spans [0, 2].
        return _clamp(1.0 - (raw_distance / 2.0))
    if metric is SimilarityMetric.DOT:
        # Oracle DOT distance is negative inner product; lower is better.
        return _sigmoid(-raw_distance)
    return 1.0 / (1.0 + max(raw_distance, 0.0))


def normalized_oracle_score(
    raw_score: float,
    metric: SimilarityMetric,
    semantics: OracleScoreSemantics,
) -> float:
    """Normalize an explicitly declared OracleVS raw-score convention."""

    if semantics is OracleScoreSemantics.DISTANCE:
        return normalized_distance(raw_score, metric)
    if metric is SimilarityMetric.EUCLIDEAN:
        raise ValueError("EUCLIDEAN OracleVS scores must be declared as DISTANCE")
    return normalized_similarity(raw_score, metric)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
