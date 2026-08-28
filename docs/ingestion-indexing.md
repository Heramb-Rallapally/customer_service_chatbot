# Ingestion to retrieval indexing

`KnowledgeIndexer` connects the existing `KnowledgeIngestionPipeline` to the
existing `RetrievalService.index_documents(documents)` batch API. It does not
create embeddings, connect to Oracle, or instantiate OracleVS. Those concerns
remain in the configured retrieval service.

```text
source file / records
       ↓
KnowledgeIngestionPipeline
       ↓  list[KnowledgeDocument]
KnowledgeIndexer.index_documents
       ↓
RetrievalService.index_documents
       ↓
configured embeddings (one OracleVS `add_texts` batch) → OracleVS
```

The indexer submits the same `KnowledgeDocument` instances it receives. Their
`id`, `content`, `source`, `product`, `version`, `issue_type`, `severity`,
`resolution_category`, and `metadata` are preserved. The retrieval adapter
serializes these fields into OracleVS metadata, including the four
conversation-filter dimensions: `product`, `version`, `issue_type`, and
`severity`. Custom metadata must be JSON serializable because the pinned
OracleVS implementation serializes it with `json.dumps`; invalid values are
rejected before the Oracle call.

For a developer-managed operation, construct the application container and run
the indexing service separately from startup:

```python
from src.app import create_application

with create_application() as application:
    application.knowledge_indexer.ingest_file_and_index("data/raw/vpn-guide.txt")
```

The application does not automatically scan, ingest, or index files at startup.
The ingestion pipeline derives deterministic document IDs. Indexing is
**insert-only**. The pinned OracleVS backend hashes supplied IDs and performs
`INSERT` into a primary-key column, so re-indexing an identical document raises
the existing retrieval/database insertion error instead of silently replacing
it. Operators must use a deliberate reset/replacement process before a
re-index; deduplication is not an indexing-side fallback.

OracleVS uses `executemany(...); commit()` for a batch but does not expose a
rollback policy through its pinned LangChain API. Partial-failure transaction
semantics require a live Oracle validation and are not claimed by this layer.
