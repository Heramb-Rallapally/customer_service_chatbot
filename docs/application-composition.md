# Application composition

`src.app.create_application()` is the runtime composition root. It creates the
production dependency graph only when called; importing `src.app` has no Oracle
Database, Ollama, or OCI credential/network/client side effect.

```text
Settings
  ├─ Oracle Database connection ──> LangChain OracleVS (COSINE)
  ├─ configured embeddings ───────┘
  │                                  ↓
  │                           OracleVSVectorStore
  │                                  ↓
  │                           RetrievalService
  ├─ configured LLM ──────────────────────────────┐
  └─ ProactiveSupportService ───────────────────────┤
       ├─ RetrievalEvidenceProvider ── RetrievalService
       └─ ConversationMemoryHistoryProvider ── memory
                                                   ↓
                                            ConversationEngine
```

The default graph uses `LLM_PROVIDER=ollama`, `EMBEDDING_PROVIDER=ollama`,
`LLM_MODEL=llama3.2:3b`, and `EMBEDDING_MODEL=nomic-embed-text` through the
local `OLLAMA_BASE_URL`. It checks the Ollama process and required models once
during lazy application construction. Oracle still requires
`ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, `ORACLE_DB_DSN`, and `ORACLEVS_TABLE`.
Set either provider to `oci` to select the retained OCI adapter; that provider
then also requires `OCI_COMPARTMENT_ID` and the appropriate OCI model ID.

OracleVS is constructed with the pinned LangChain Community API and
`DistanceStrategy.COSINE`; the hardened `OracleVSVectorStore` performs the
project's score conversion and filtering. OracleVS creates the configured table
when needed. Vector-index provisioning remains a deliberate database deployment
operation and is not performed automatically at application startup.

`ApplicationServices.close()` closes an Oracle connection created by the
container. An injected connection remains caller-owned. The default
`InMemoryConversationMemory` is thread-safe but process-local, so it is only
appropriate for local development or a single application process.

The default proactive graph uses the existing `RetrievalService` for
evidence-backed recommendations and unsupported-issue assessment, and the
configured conversation memory for user-isolated history. Its default sentiment
analyzer is deterministic and local. Hosts may inject `OciSentimentAnalyzer`
with a reviewed OCI-backed callable through `proactive_sentiment_analyzer`; the
proactive package does not create a second OCI client at import time.

Tests may inject `retrieval_service`, `llm_service`, memory, proactive
services, the Ollama API client, or individual proactive providers. Fully
injected tests make no model-provider or Oracle connection.

## Optional analytics and feedback

Analytics is an observer of the application, not a conversation dependency.
`ApplicationServices` owns an injected `AnalyticsEventSink`; the default is a
`NoOpAnalyticsEventSink`, so normal chat remains available if analytics is not
configured. Set `ANALYTICS_MODE=memory` only for a local demo to use the
thread-safe, process-local `InMemoryAnalyticsEventSink`. It is not durable or
shared across workers.

After a successful chat, `ChatApplicationService` records a typed
`SupportEvent` containing outcome metadata such as resolution status,
confidence, response timing, citations, and suggested-action count. It never
stores raw customer or assistant messages. `POST /feedback` creates a separate
helpfulness event after checking the authenticated user's conversation
ownership. Analytics failures are logged as sanitized operational warnings and
never fail a chat or feedback acknowledgement.

`GET /analytics/events` is intentionally scoped to the authenticated user's
own event snapshots for the local Streamlit view. It is not an administrative
reporting API. `src.analytics.to_evaluation_records()` converts outcome and
feedback metadata into dependency-free offline evaluation records for Step 8;
it performs neither model retraining nor prompt modification.
