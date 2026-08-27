"""Similarity metrics supported by retrieval vector-store adapters."""

from __future__ import annotations

from enum import Enum


class SimilarityMetric(str, Enum):
    """Vector-comparison strategies accepted by the retrieval layer."""

    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"
    DOT = "DOT"
