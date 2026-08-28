# OracleVS retrieval integration

The retrieval adapter is implemented and tested against these exact package versions:

- `langchain-community==0.3.31`
- `langchain==0.3.30`
- `langchain-core==0.3.86`
- `oracledb==3.4.2`
- `oci==2.168.3` (retained for optional OCI provider selection)

## Actual OracleVS API

Construct the LangChain store with `OracleVS(client, embedding_function,
table_name, distance_strategy, query, params)`. `client` is an `oracledb`
connection or pool. Construction creates a table with `id`, `text`,
`metadata`, and `embedding vector(<embedding dimension>, FLOAT32)` when the
table does not already exist. Create an index with
`langchain_community.vectorstores.oraclevs.create_index(client, vector_store,
params)`. The supported insertion API is `add_texts(texts, metadatas, ids)`;
the pinned release has no `add_embeddings` method. The supported vector search
API is `similarity_search_by_vector_with_relevance_scores(embedding, k, filter)`;
there is no `similarity_search_with_score_by_vector` method.

OracleVS 0.3.31 implements that search with `vector_distance(...) AS distance`
and `ORDER BY distance`, so its returned scalar is a distance and lower is
better. Its native metadata filtering checks `metadata.get(key) in value`,
which is not safe exact matching for scalar string filters. The project thus
does not pass filters to OracleVS. It progressively doubles an unfiltered
candidate request, applies normalized exact matching locally, and stops only
when it finds `top_k` matches or the backend returns fewer candidates than
requested. An explicitly configured `max_candidate_fetch` fails visibly rather
than returning a silently incomplete filtered result.

## Production configuration

Production metric is **COSINE**. Configure both `OracleVS(distance_strategy=
DistanceStrategy.COSINE)` and its vector index for `COSINE`, then construct
`OracleVSVectorStore(..., metric=SimilarityMetric.COSINE, embedding_dimension=<model dimension>)`.
The adapter validates injected OracleVS strategy when exposed, document/query
vector dimensions, non-empty vectors, and finite values. Set
`EMBEDDING_DIMENSION` to the selected model's documented output dimension.
The default local `nomic-embed-text` model returns 768 values.
`OllamaEmbeddingService` and the retained `OCIEmbeddingService` validate every
provider response against the configured dimension and are LangChain
`Embeddings` implementations, so either can be passed directly as
OracleVS's `embedding_function`. Supply the same dimension to
`OracleVSVectorStore` during production wiring.

OracleVS creates `embedding VECTOR(<dimension>, FLOAT32)` only when the table
does not already exist. Oracle 23ai Free does not expose vector dimensions in
`USER_TAB_COLUMNS`, so the composition root inspects the existing table DDL
through `DBMS_METADATA.GET_DDL` before initializing OracleVS and fails if the
fixed column dimension differs from the selected embedding provider. A vector index does not have
a separate embedding dimension; it indexes that fixed-dimension column. Never
mix vectors from OCI and Ollama models in one table, even if their dimensions
happen to match. Switching embedding models requires a new/reset table and a
complete knowledge re-index. Schema replacement remains an operator-managed
deployment action.

Score conversion is stable across queries and never uses per-query min-max normalization:

| Metric | Raw OracleVS value | Shared relevance score | Production status |
| --- | --- | --- | --- |
| COSINE | cosine distance, lower is better, expected `[0, 2]` | `clamp(1 - distance / 2)` | Supported and production-safe |
| EUCLIDEAN | Euclidean distance, lower is better | `1 / (1 + distance)` | Supported; benchmark-only unless explicitly configured end-to-end |
| DOT | negative dot-product distance, lower is better | `sigmoid(-distance)` | Supported; benchmark-only unless explicitly configured end-to-end |

All final scores are finite values in `[0, 1]`, higher is better, and are
sorted descending before returning `RetrievalResult` values.

## Live integration test

`tests/integration/test_oraclevs_live.py` is intentionally skipped unless all
of the following are set: `RUN_ORACLEVS_INTEGRATION=1`, `ORACLE_DB_USER`,
`ORACLE_DB_PASSWORD`, `ORACLE_DB_DSN`, and `ORACLEVS_INTEGRATION_TABLE`.
It uses the real pinned OracleVS class and database driver, inserts and searches
an isolated caller-provided table, then removes its test rows. It does not run
without an Oracle AI Database 23ai environment and does not claim that a mock
verifies OracleVS.
