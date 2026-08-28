# Retrieval

`RetrievalService` is the conversation-facing knowledge-search boundary:

```python
search(*, query: str, filters: Mapping[str, str], top_k: int) -> Sequence[RetrievalResult]
```

It returns shared `RetrievalResult` objects with document ID, content, metadata,
and a finite relevance score from `0.0` to `1.0`, where higher is better.

## Search flow

```text
query → embedding provider → OracleVS vector search → local metadata filtering
      → score conversion → descending RetrievalResult values
```

The production metric is COSINE. With the pinned
`langchain-community==0.3.31` OracleVS implementation, the method named
`similarity_search_by_vector_with_relevance_scores` returns Oracle vector
**distance** values, ordered low to high. The adapter converts COSINE distance
to relevance with `clamp(1 - distance / 2)`, then sorts final results by
descending relevance.

Other adapter metrics are available for benchmarks: Euclidean uses
`1 / (1 + distance)` and dot product uses `sigmoid(-distance)`. Production
wiring is COSINE end-to-end.

## Filtering

Supported conversation filter keys are:

- `product`
- `version`
- `issue_type`
- `severity`

OracleVS 0.3.31 does not provide safe scalar exact filtering for this use case.
The adapter retrieves unfiltered candidates, normalizes values, and applies
exact local filtering while progressively increasing candidate count. It either
returns up to `top_k` matches or fails clearly if an explicitly configured
candidate limit is reached.

## OracleVS API compatibility

This project is implemented against:

- `langchain-community==0.3.31`
- `langchain==0.3.30`
- `langchain-core==0.3.86`
- `oracledb==3.4.2`

The supported OracleVS insertion API is `add_texts(texts, metadatas, ids)`.
There is no `add_embeddings` method in this pinned version. OracleVS embeds
indexed text through its configured embedding function, so the production
indexing path embeds each document once.

## Safety and model changes

The adapter validates non-empty, numeric, finite embedding vectors and their
dimension. It also validates JSON-serializable metadata before insertion. Do
not mix embeddings from different models in one Oracle vector table; create or
reset a table and re-index the full corpus when changing embedding models.

See [Oracle Database 23ai](oracle-23ai.md) for table and index requirements.
