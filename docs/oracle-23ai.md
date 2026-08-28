# Oracle Database 23ai and OracleVS

Oracle Database 23ai is the only vector database used by this project. LangChain
OracleVS stores chunk text, JSON metadata, and an Oracle `VECTOR` column; the
application retains Oracle for both retrieval and optional durable conversation
memory.

## Vector table requirements

The default `nomic-embed-text` embedding model produces 768-dimensional
vectors. Configure the knowledge table used by `ORACLEVS_TABLE` for:

```text
EMBEDDING VECTOR(768, FLOAT32)
```

Use an HNSW vector index configured for COSINE distance. The vector index and
the OracleVS distance strategy must both use COSINE.

On first initialization, the pinned OracleVS implementation can create a
missing table with its `id`, `text`, `metadata`, and `embedding` columns. It
does not provision the HNSW index automatically. Create the index through your
approved Oracle schema/deployment process, then verify it against your Oracle
23ai deployment standards.

The composition root reads existing table DDL before OracleVS initialization
and fails if the `EMBEDDING` dimension differs from `EMBEDDING_DIMENSION`.
This prevents opaque database errors and accidental use of a stale table.

## Embedding compatibility

Dimension equality is necessary but not sufficient. Vectors from different
embedding models occupy different semantic spaces even when they have the same
dimension. Never mix OCI and Ollama vectors, or two model versions, in the same
table. Use a new or intentionally reset table and re-index all documents after
an embedding-model change.

## Conversation memory table

Set `ORACLE_CONVERSATION_TABLE` only after provisioning
[`scripts/oracle-conversation-memory.sql`](../scripts/oracle-conversation-memory.sql). The table
stores the validated JSON `ConversationState`, user ownership, a summary,
timestamps, and a version field.

`OracleConversationMemory` performs optimistic updates:

```text
load version N → process turn → UPDATE ... WHERE version = N → version N+1
```

A concurrent write raises a typed conflict instead of silently overwriting a
newer turn. The application does not create or alter this table at startup.

## Connection settings

Use a normal Oracle DSN such as `localhost:1521/FREEPDB1` and provide
`ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, and `ORACLE_DB_DSN` locally. Keep those
values out of version control. For live retrieval, the configured user needs
the privileges required to create/use the chosen table and index.
