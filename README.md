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

## Run the API and UI

Install dependencies, configure the OCI/Oracle values in your environment, then
start the API from the repository root:

```bash
uvicorn src.api.app:app --reload
```

The API exposes `GET /health` for process health and `POST /chat` using the
existing `ChatRequest` and `ChatResponse` models. OCI/Oracle clients are
created lazily on the first chat request through `src.app.create_application()`;
importing the API or calling `/health` does not require infrastructure.

Start Streamlit in a second terminal:

```bash
streamlit run src/ui/app.py
```

The UI communicates only through FastAPI using `HttpChatApiClient`. Set
`API_BASE_URL` when the API is not at `http://127.0.0.1:8000`. Real chat
requires configured OCI Generative AI and Oracle AI Database access; those
services are not exercised by the credential-free test suite.

## Analytics and feedback

Analytics observes completed support outcomes; it does not control retrieval,
conversation, or generation. The API records typed `SupportEvent` metadata
(resolution/escalation state, confidence, timing, citation/action counts and
optional feedback) without storing raw chat text. `POST /feedback` accepts
`conversation_id`, a `positive` or `negative` rating, and an optional comment.
Its user identity always comes from `AuthenticatedIdentity`; feedback for a
different user's conversation receives the same safe 403 ownership response as
chat.

`ANALYTICS_MODE=noop` is the default. Set `ANALYTICS_MODE=memory` for a local,
single-process demo to enable the Streamlit **Your support activity** view via
the API's authenticated, user-scoped `GET /analytics/events` endpoint. This
in-memory sink is not durable and is not a production analytics store. Event
metadata can be transformed into offline Step 8 evaluation records, but Step 7
does not retrain models or change prompts automatically.

## API identity and conversation ownership

`POST /chat` now derives its effective `user_id` from an injected
`AuthenticatedIdentity`, not from `ChatRequest.user_id`. The request field is
retained for compatibility only: if supplied, it must match the authenticated
identity or the API returns a safe 403 response. It cannot impersonate another
user.

For local demos, `API_AUTH_MODE=development` uses the server-side
`API_DEVELOPMENT_USER_ID` value (default: `local-demo-user`). The UI/client may
omit `user_id`; it is not authentication and does not establish identity.

For production, set `API_AUTH_MODE=required` and inject authentication
middleware that verifies the deployment's trusted session/token mechanism and
sets `request.state.authenticated_identity` to `AuthenticatedIdentity`. Without
that middleware the API returns 401. Conversation ownership then ensures the
authenticated user cannot continue another user's conversation.

## Conversation memory

`InMemoryConversationMemory` is process-local and intended only for tests and
local development. Configure `ORACLE_CONVERSATION_TABLE` after provisioning the
documented Oracle schema to select durable, optimistic-concurrency memory in
the application composition root. See [conversation memory documentation](docs/conversation-memory.md).

## Proactive providers

The application composition root wires retrieval-backed proactive
recommendations, evidence-based unsupported-issue detection, and
user-isolated conversation history. See [proactive provider documentation](docs/proactive-providers.md)
for dependency, graceful-degradation, and OCI sentiment-injection details.
