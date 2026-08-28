"""Unit tests for dependency-light evaluation contracts, metrics, and reports."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationDatasetError,
    RecordingRetriever,
    load_dataset,
    render_console_report,
    report_json,
)
from src.api import ChatApplicationService
from src.evaluation import EvaluationRunner
from src.evaluation.run import main
from src.models import ChatResponse, ResolutionStatus, RetrievalResult


def case(**updates: object) -> EvaluationCase:
    values: dict[str, object] = {
        "case_id": "case-1",
        "user_message": "My Oracle VPN version 5.2 reports authentication failed.",
        "expected_resolution_status": ResolutionStatus.AWAITING_CONFIRMATION,
        "expected_escalation": False,
    }
    values.update(updates)
    return EvaluationCase(**values)


def test_dataset_contract_loads_json_and_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(EvaluationDataset(cases=[case()]).model_dump(mode="json")),
        encoding="utf-8",
    )
    loaded = load_dataset(path)
    assert loaded.cases[0].case_id == "case-1"

    with pytest.raises(ValidationError, match="unique"):
        EvaluationDataset(cases=[case(), case()])


def test_dataset_loader_fails_safely_for_invalid_content(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="not valid JSON"):
        load_dataset(path)
    assert main(["--dataset", str(tmp_path / "missing.json")]) == 2


def test_case_contract_rejects_blank_messages_and_invalid_expected_status() -> None:
    with pytest.raises(ValidationError):
        case(user_message=" ")
    with pytest.raises(ValidationError):
        case(expected_resolution_status="NOT_A_STATUS")


def test_recording_retriever_delegates_once_and_returns_defensive_results() -> None:
    class Retriever:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, **_kwargs: object) -> list[RetrievalResult]:
            self.calls += 1
            return [
                RetrievalResult(
                    document_id="doc-1",
                    content="Supported guidance",
                    score=0.9,
                    metadata={"source": "guide"},
                )
            ]

    delegate = Retriever()
    recorder = RecordingRetriever(delegate)
    checkpoint = recorder.checkpoint()
    results = recorder.search(query="vpn", filters={}, top_k=5)
    observations = recorder.observations_since(checkpoint)

    assert delegate.calls == 1
    assert results[0].document_id == "doc-1"
    assert len(observations) == 1
    assert observations[0].results[0] is not results[0]


def test_report_renderers_never_require_missing_metrics(local_evaluation_report) -> None:
    report = local_evaluation_report
    serialized = json.loads(report_json(report))
    console = render_console_report(report)

    assert set(serialized) == {"summary", "metrics", "breakdown", "cases"}
    assert "Cases evaluated:" in console
    assert "Failed cases" in console


def test_missing_state_analytics_and_retrieval_metrics_remain_unavailable() -> None:
    class Conversation:
        def handle_message(self, **_kwargs: object) -> ChatResponse:
            return ChatResponse(message="Safe response")

    class MissingState:
        def get_state(self, _conversation_id: str):
            return None

    runner = EvaluationRunner(
        ChatApplicationService(Conversation()),
        state_reader=MissingState(),
        conversation_id_factory=lambda evaluation_case: evaluation_case.case_id,
    )
    report = runner.run(EvaluationDataset(cases=[case()]))

    assert report.metrics.resolution_rate is None
    assert report.metrics.resolved_cases is None
    assert report.metrics.unresolved_cases is None
    assert report.metrics.average_confidence is None
    assert report.metrics.average_response_time_ms is None
    assert report.metrics.retrieval_hit_rate is None
    assert report.cases[0].actual_resolution_status is None
    assert report.cases[0].failure_reasons == ["resolution_status_mismatch"]
    assert "Safe response" not in report_json(report)
    assert "resolution_status_mismatch" in render_console_report(report)
