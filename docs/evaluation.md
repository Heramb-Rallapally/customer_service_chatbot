# Evaluation

The evaluation tooling evaluates the existing support pipeline without changing
production conversation behavior. The runner submits each labelled case through
`ChatApplicationService`, reads the state persisted by `ConversationEngine`,
consumes `SupportEvent` metadata, and can observe the retriever through a
transparent `RecordingRetriever` wrapper.

```text
EvaluationDataset
      ↓
EvaluationRunner
      ↓
ChatApplicationService
      ↓
ConversationEngine
  ├─ ProactiveSupportService
  ├─ Retriever → OracleVS when live
  ├─ LLMService → configured Ollama or OCI adapter when live
  └─ ConversationMemory
      ↓
ChatResponse + ConversationState + SupportEvent
      ↓
EvaluationReport (summary, metrics, breakdown, cases)
```

The runner never calls the LLM or vector store directly. `RecordingRetriever`
delegates each search exactly once and stores defensive result copies, so
evaluation does not create an extra embedding or retrieval call. Standalone
Recall@K and MRR evaluation remains in `src/retrieval/evaluation.py`; run that
separately when a labelled query-to-document dataset is available.

## Dataset format

The JSON contract is versioned and contains unique cases. Each case requires:

- `case_id`
- `user_message`
- `expected_resolution_status`
- `expected_escalation`

Optional fields include `issue_type`, `category`, `difficulty`,
`follow_up_messages`, `expected_keywords`, `expected_source_ids`, an evaluation
`feedback_rating`, and extensible metadata. Expected keyword matching is
case-insensitive and requires all configured keywords. Source matching requires
all configured identifiers to appear as citation document IDs/sources or
related-article identifiers/sources/titles.

See `examples/evaluation/cases.json` for a small labelled sample. Expected
source identifiers must be adjusted to identifiers actually present in the
target knowledge index.

## Running against configured model and Oracle services

From the repository root:

```bash
python -m src.evaluation.run \
  --dataset examples/evaluation/cases.json \
  --output examples/evaluation/results.json
```

Use `--json` to print JSON instead of the readable console report. The output
contract contains `summary`, `metrics`, `breakdown`, and `cases`. It deliberately
does not contain raw user messages, prompts, generated responses, credentials,
DSNs, or provider errors.

The CLI uses `create_application()`, so a live run requires the selected model
provider and the same Oracle configuration as the production application.
Configuration and client creation failures return a non-zero exit status. Live
model-provider/Oracle execution is not part of the normal test suite.

## Credential-free local evaluation

Tests inject deterministic implementations of `Retriever`, `LLMService`,
`ConversationMemory`, `ProactiveService`, and `AnalyticsEventSink` into the real
composition root. This exercises ingestion/indexing, `ConversationEngine`, the
application service, memory, analytics, and evaluation without external calls.
Use `InMemoryConversationMemory` and `InMemoryAnalyticsEventSink` only for local
evaluation; neither is durable or shared across processes.

To capture retrieval metrics without a second search, wrap the injected
retriever before composition:

```python
recording_retriever = RecordingRetriever(retriever)
services = create_application(
    retrieval_service=recording_retriever,
    llm_service=llm,
    proactive_service=proactive,
    memory=memory,
    analytics_sink=analytics,
)
runner = EvaluationRunner(
    ChatApplicationService(services.conversation_engine, analytics_sink=analytics),
    state_reader=services.conversation_engine,
    analytics_source=analytics,
    retrieval_observer=recording_retriever,
)
```

## Metrics

Reports include pass, resolution, escalation, confidence, response-time,
citation, suggested-action, retrieval-hit, retrieval-score, and labelled
expected-vs-actual metrics. Breakdowns cover issue type, final resolution
status, escalation status, and retrieval quality. Per-case results contain only
outcomes and failure reason codes.

Rates use only values that are actually available. Missing confidence, timing,
state, analytics, or retrieval observations remain `null`/`unavailable`; they
are not converted into zero. Response time is the final evaluated turn's
chat-operation timing. Retrieval score is the highest observed score across
the case's real retrieval calls.

Evaluation does not retrain a model, modify prompts, update production
knowledge, or change runtime decisions. Those actions require separate human
review. Analytics remains best-effort, so evaluation can still report response
and state outcomes when an analytics sink fails.
