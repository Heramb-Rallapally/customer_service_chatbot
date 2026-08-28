# Proactive providers

`ProactiveSupportService` remains the only provider-facing service used by the
Conversation Engine. Optional provider failures are reduced to absent evidence
or `UNKNOWN` sentiment, so they do not stop grounded response generation.

## Providers

- `RuleBasedSentimentAnalyzer` is deterministic and used by default for local
  development and credential-free tests.
- `OciSentimentAnalyzer` is production-capable when a hosting environment
  injects a reviewed OCI-backed callable. It normalizes only the shared
  `Sentiment` values and returns `UNKNOWN` for invalid output or failures. It
  does not create OCI clients or load credentials itself.
- `RetrievalEvidenceProvider` uses the existing `Retriever.search` contract.
  It creates no embeddings and does not access OracleVS directly. One bounded
  cache entry is reused by unsupported detection, related articles, and similar
  issue references for the same message/context. Article IDs, titles, and
  sources are copied only from `RetrievalResult` evidence.
- `ConversationMemoryHistoryProvider` uses `ConversationMemory.list_for_user`
  and requires the `ConversationState.user_id` already established from the
  authenticated API identity. It verifies returned records have the same owner
  before creating references; it never queries Oracle directly.

## Support assessment

Retrieval evidence is interpreted as `SUPPORTED` (score at least 0.75),
`POTENTIALLY_SUPPORTED` (weak but non-empty evidence), `UNSUPPORTED` (no
evidence), or `UNAVAILABLE` (retrieval infrastructure failure). Only
`UNSUPPORTED` triggers the existing boolean unsupported-issue escalation path;
an infrastructure failure is not treated as an unsupported customer issue.

## Production notes

Configure durable memory as documented in [conversation-memory.md](conversation-memory.md)
to retain history across processes. Production OCI sentiment integration is
host-injected; live OCI/Oracle provider behavior is not exercised by the
credential-free tests.
