# Application composition

`src.app.create_application()` is the runtime composition root. It creates the
production dependency graph only when called; importing `src.app` has no Oracle
Database or OCI credential, network, or client side effect.

```text
Settings
  ├─ Oracle Database connection ──> LangChain OracleVS (COSINE)
  ├─ OCI embeddings ──────────────┘
  │                                  ↓
  │                           OracleVSVectorStore
  │                                  ↓
  │                           RetrievalService
  ├─ OCI Cohere LLM ──────────────────────────────┐
  └─ local ProactiveSupportService + memory ───────┤
                                                   ↓
                                            ConversationEngine
```

The production graph requires `OCI_COMPARTMENT_ID`, `EMBEDDING_MODEL`,
`LLM_MODEL`, `ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, `ORACLE_DB_DSN`, and
`ORACLEVS_TABLE`. `ORACLEVS_TABLE` is validated as a single Oracle identifier.
`OCI_CONFIG_PROFILE` and `OCI_ENDPOINT` remain optional configuration inputs
used by the OCI SDK clients.

OracleVS is constructed with the pinned LangChain Community API and
`DistanceStrategy.COSINE`; the hardened `OracleVSVectorStore` performs the
project's score conversion and filtering. OracleVS creates the configured table
when needed. Vector-index provisioning remains a deliberate database deployment
operation and is not performed automatically at application startup.

`ApplicationServices.close()` closes an Oracle connection created by the
container. An injected connection remains caller-owned. The default
`InMemoryConversationMemory` is thread-safe but process-local, so it is only
appropriate for local development or a single application process.

Tests may inject `retrieval_service`, `llm_service`, memory, and proactive
services. Fully injected tests make no OCI or Oracle connection.
