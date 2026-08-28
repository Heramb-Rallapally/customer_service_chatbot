"""JSON evaluation dataset loading with explicit, safe validation failures."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import EvaluationDataset


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be loaded or validated."""


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load a UTF-8 JSON dataset without executing or importing its contents."""

    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationDatasetError(
            f"Unable to read evaluation dataset: {dataset_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset is not valid JSON: {dataset_path}"
        ) from exc
    try:
        return EvaluationDataset.model_validate(payload)
    except ValidationError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset does not match the required schema: {dataset_path}"
        ) from exc
