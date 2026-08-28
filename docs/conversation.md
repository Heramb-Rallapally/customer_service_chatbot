# Conversation, grounding, and memory

`ConversationEngine` coordinates message understanding, state retention,
proactive signals, retrieval, grounded generation, citations, and resolution
status. Its public turn API is synchronous:

```python
handle_message(
    *, conversation_id: str, user_message: str, user_id: str | None = None
) -> ChatResponse
```

At the HTTP boundary, the effective user identity comes from
`AuthenticatedIdentity`, not the request body. The engine rejects attempts to
continue a user-bound conversation with a missing or different user ID.

## Turn lifecycle

```text
user message
  → intent and context update
  → optional proactive analysis / deterministic escalation
  → clarification or retrieval
  → grounded LLM generation
  → engine-built citations and response
  → state persistence
```

The engine retains recent messages, product/version/issue context, failed and
suggested troubleshooting steps, resolution status, and ownership in a
`ConversationState`.

## Clarification policy

Structured product, version, and issue fields improve targeted troubleshooting,
but they are not mandatory for every question.

- **Self-contained information question:** “What is Oracle AI Database at AWS
  and which AWS regions are supported?” proceeds directly to retrieval.
- **Genuinely ambiguous request:** “How do I fix this?” without preceding
  context can receive a targeted clarification question.
- **Known context:** once a product or version is supplied, the engine avoids
  asking for that same field again.

If retrieval is empty, weak, unavailable, or unsupported, the engine uses a
safe fallback/escalation instead of making up support guidance.

## Grounding and citations

The LLM receives explicitly delimited system instructions, user data,
conversation context, excluded failed steps, and retrieved knowledge. User and
retrieved text are untrusted data, not instructions. The prompt requires a
direct answer when evidence is sufficient, requires uncertainty when a requested
fact is absent, and prohibits invented facts, document IDs, and citations.

The LLM returns only a message, suggested actions, and confidence. The engine
creates each `Citation` from actual `RetrievalResult` metadata (`source`, then
`title`, then document ID), so a model cannot fabricate a source.

## Resolution states

Typical statuses include `NEEDS_CLARIFICATION`, `READY_TO_RESOLVE`,
`AWAITING_CONFIRMATION`, `RESOLVED`, and `ESCALATED`.

- A successful factual answer can be `RESOLVED` immediately.
- Troubleshooting guidance normally becomes `AWAITING_CONFIRMATION` until the
  customer confirms it worked.
- Explicit human requests, repeated failed steps, critical severity, high
  frustration, or insufficient/unsupported knowledge can escalate.

The FastAPI response exposes the final state as the `X-Resolution-Status`
header while `ChatResponse.message` remains the customer-facing answer.

## Memory options

`InMemoryConversationMemory` is thread-safe but process-local. It is suitable
for tests and a single local process only.

Set `ORACLE_CONVERSATION_TABLE` to use `OracleConversationMemory`. It persists
validated JSON state in Oracle and uses optimistic concurrency, preventing a
stale request from overwriting a newer turn. Provision the schema before
enabling it; see [Oracle 23ai](oracle-23ai.md).

## Proactive support

`ProactiveSupportService` is optional and cannot break a normal chat turn when
an optional provider fails. Its retrieval-evidence provider reuses the existing
retriever for related articles and unsupported-issue assessment, avoiding a
second vector database or embedding pipeline. Customer history reads only the
authenticated user's conversation memory.
