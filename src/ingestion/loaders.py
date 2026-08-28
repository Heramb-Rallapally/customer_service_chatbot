"""Local source loaders for common knowledge-export formats."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .exceptions import IngestionError, UnsupportedSourceError
from .models import IngestionRecord, SourceType

_TEXT_SUFFIXES = {".txt", ".md", ".rst", ".vtt", ".srt"}
_RESERVED_FIELDS = {"content", "text", "body", "source", "source_type", "metadata"}


def load_file(path: str | Path, source_type: SourceType | None = None) -> list[IngestionRecord]:
    """Load text, JSON, or CSV knowledge exports without external services."""

    source_path = Path(path)
    if not source_path.is_file():
        raise IngestionError(f"source file does not exist: {source_path}")

    resolved_type = source_type or infer_source_type(source_path)
    suffix = source_path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return [IngestionRecord(source=str(source_path), content=source_path.read_text(encoding="utf-8"), source_type=resolved_type)]
    if suffix == ".json":
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestionError(f"invalid JSON source: {source_path}") from exc
        entries = payload if isinstance(payload, list) else [payload]
        return [record_from_mapping(entry, str(source_path), resolved_type) for entry in entries]
    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            return [record_from_mapping(row, str(source_path), resolved_type) for row in csv.DictReader(handle)]
    raise UnsupportedSourceError(f"unsupported source format: {suffix or 'no extension'}")


def record_from_mapping(
    entry: Mapping[str, Any], default_source: str, default_source_type: SourceType
) -> IngestionRecord:
    """Create a record from a JSON object or CSV row and retain non-core fields."""

    if not isinstance(entry, Mapping):
        raise IngestionError("structured source entries must be objects")
    content = entry.get("content", entry.get("text", entry.get("body")))
    if not isinstance(content, str):
        raise IngestionError("structured source entry requires string content, text, or body")
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise IngestionError("metadata must be an object")
    merged_metadata = {**metadata, **{key: value for key, value in entry.items() if key not in _RESERVED_FIELDS}}
    raw_type = entry.get("source_type", default_source_type)
    try:
        resolved_type = raw_type if isinstance(raw_type, SourceType) else SourceType(raw_type)
    except ValueError as exc:
        raise IngestionError(f"unknown source type: {raw_type}") from exc
    return IngestionRecord(
        source=str(entry.get("source") or default_source),
        content=content,
        source_type=resolved_type,
        metadata=merged_metadata,
    )


def infer_source_type(path: Path) -> SourceType:
    """Infer a conservative source category from a filename when none is supplied."""

    name = path.stem.lower()
    if "faq" in name:
        return SourceType.FAQ
    if "ticket" in name:
        return SourceType.HISTORICAL_TICKET
    if "knowledge" in name or "kb" in name:
        return SourceType.KNOWLEDGE_BASE
    if "troubleshoot" in name or "guide" in name:
        return SourceType.TROUBLESHOOTING_GUIDE
    if "transcript" in name or path.suffix.lower() in {".vtt", ".srt"}:
        return SourceType.VIDEO_TRANSCRIPT
    return SourceType.PRODUCT_DOCUMENTATION
