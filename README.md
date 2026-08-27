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
