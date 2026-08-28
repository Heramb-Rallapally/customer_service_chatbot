# Ingestion and indexing

The project indexes knowledge through the existing ingestion and retrieval
contracts; it has no standalone ingestion CLI.

```text
data/raw/official_docs
        ↓
file loading → cleaning → metadata extraction → chunking
        ↓
list[KnowledgeDocument]
        ↓
KnowledgeIndexer
        ↓
RetrievalService.index_documents
        ↓
Ollama nomic-embed-text → OracleVS → Oracle Database 23ai
```

## Supported source formats

`KnowledgeIngestionPipeline` supports plain text (`.txt`, `.md`, `.rst`),
subtitle files (`.vtt`, `.srt`), JSON, and CSV. JSON/CSV records require a
string `content`, `text`, or `body` field. Source type is inferred from the
file name when not supplied.

## Index one file

Load your environment first, then invoke the composition root and indexer:

```bash
set -a
source .env
set +a

python - <<'PY'
from src.app import create_application

with create_application() as application:
    documents = application.knowledge_indexer.ingest_file_and_index(
        "data/raw/official_docs/example.txt"
    )
    print(f"Indexed {len(documents)} chunks")
PY
```

Replace the example path with an existing supported file. The same approach can
iterate over `data/raw/official_docs/*.txt`; see the repository README for a
corpus loop.

## Metadata and identifiers

Each chunk is a `KnowledgeDocument` with a deterministic ID, content, source,
and structured metadata including product, version, issue type, severity,
resolution category, source type, chunk index, and chunk count. Custom metadata
must be JSON serializable because OracleVS stores it as JSON.

The conversation retrieval filters are `product`, `version`, `issue_type`, and
`severity`. Preserve those names when adding structured source metadata.

## Re-indexing and failures

Indexing is insert-only. Re-indexing the same chunk IDs can trigger a duplicate
key error from Oracle; the application does not silently replace documents.
Use a deliberate new/reset table and complete re-index when rebuilding a corpus
or changing embedding models.

The indexing adapter delegates embeddings to OracleVS's supported `add_texts`
path, so documents are embedded once. Retrieval, embedding, and Oracle failures
propagate rather than being converted into a successful empty index.
