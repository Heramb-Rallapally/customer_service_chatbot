# Customer Service Chatbot

A team capstone project for a retrieval-augmented customer support assistant with multi-turn resolution. This repository currently contains only the shared project foundation: data contracts, environment configuration, package boundaries, and contract tests.

## Setup

Python 3.9 or newer is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Real credentials are not required to import the project or run its contract tests. `.env.example` documents environment variable names for later integrations; no `.env` file is loaded automatically.

Run the test suite with:

```bash
python -m pytest
```

## Retrieval module

`src/retrieval/` exposes a mock-friendly `RetrievalService`, metadata filters,
and deterministic in-memory implementations for local development. Its public
conversation-facing API is
`search(*, query: str, filters: Mapping[str, str], top_k: int)`. It supports
COSINE, EUCLIDEAN, and DOT vector metrics; all returned scores are normalized
to `[0, 1]`, where higher is better. Production
integration uses `OCIEmbeddingService` and an injected LangChain OracleVS
instance through `OracleVSVectorStore`; database connection and schema setup
remain owned by the database module. OCI use requires `OCI_COMPARTMENT_ID`,
`EMBEDDING_MODEL`, and a valid OCI profile. Retrieval evaluation provides
Recall@K and MRR against explicit query-to-document relevance labels.

Oracle production integration is pinned to `langchain-community==0.3.31`,
`langchain==0.3.30`, `langchain-core==0.3.86`, and `oracledb==3.4.2`.
`OracleVSVectorStore` uses that release's `add_texts` and
`similarity_search_by_vector_with_relevance_scores` APIs. The latter returns
Oracle `vector_distance` values (lower is better), despite its method name.
The adapter converts them to bounded, higher-is-better relevance scores before
they reach the conversation layer. See `docs/oraclevs-integration.md` for the
metric, filter, and live-integration-test assumptions.

## Repository layout and ownership

- `src/models/`: shared, dependency-light Pydantic contracts used by all modules.
- `src/config.py`: centralized environment configuration without service-client initialization.
- `src/ingestion/` and `src/db/`: Member 1, knowledge ingestion and database access.
- `src/retrieval/`: Member 2, retrieval and RAG infrastructure.
- `src/conversation/`: Member 3, conversation orchestration.
- `src/proactive/`: Member 4, proactive support signals.
- `src/api/` and `src/ui/`: Member 5, API, UI, analytics, and integration.
- `tests/`: unit, integration, and evaluation tests.
- `data/`: raw, processed, and sample data locations.

See `AGENTS.md` for the authoritative architecture, integration, security, and team workflow rules.
