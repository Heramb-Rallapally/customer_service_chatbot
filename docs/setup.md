# Setup

These steps prepare a clean checkout for the default local stack: Ollama plus
Oracle Database 23ai.

## Prerequisites

- Python 3.9 or later.
- `pip` and a virtual-environment tool.
- Ollama installed and running locally.
- Oracle Database 23ai reachable from this machine, with a user permitted to
  create/use the configured vector table and durable-memory table.

## Install Python dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If imports fail with `ModuleNotFoundError: No module named 'src'`, confirm the
current directory is the repository root. If needed:

```bash
export PYTHONPATH="$PWD"
```

## Install and verify Ollama

Install Ollama using the instructions for your platform, then start its local
server if it is not already managed by the platform:

```bash
ollama serve
```

In another terminal, pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

After loading configuration below, verify the application can reach the local
API and find the selected models:

```bash
python -m src.ollama.health
```

## Configure environment variables

Copy the tracked template, then edit the copy locally:

```bash
cp .env.example .env
```

Set at least the following values in `.env`:

```dotenv
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=120
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

ORACLE_DB_USER=<your_user>
ORACLE_DB_PASSWORD=<your_password>
ORACLE_DB_DSN=localhost:1521/FREEPDB1
ORACLEVS_TABLE=<your_vector_table>
ORACLE_CONVERSATION_TABLE=<your_conversation_table>
```

The application intentionally does not parse `.env`. Load it before commands:

```bash
set -a
source .env
set +a
```

Never commit `.env`, database passwords, tokens, or OCI credentials.

## Prepare Oracle Database 23ai

The default embedding model produces 768-dimensional vectors. The OracleVS
table therefore must use `VECTOR(768, FLOAT32)` and a COSINE-compatible vector
index. See [Oracle Database 23ai](oracle-23ai.md) for the schema and index
requirements.

Create or select an application schema user according to your Oracle security
policy. A DBA-managed local development example is:

```sql
-- Run by a DBA. Replace placeholders and apply your organization's password,
-- tablespace, quota, and least-privilege policies.
CREATE USER <app_user> IDENTIFIED BY "<strong_password>";
GRANT CREATE SESSION, CREATE TABLE, CREATE INDEX TO <app_user>;
ALTER USER <app_user> QUOTA UNLIMITED ON <tablespace_name>;
```

The user also needs access to any Oracle metadata/package features required by
your database policy for schema validation. Prefer a DBA-provisioned schema in
shared or production environments rather than running this example unchanged.

OracleVS can create a missing vector table during application construction.
The application does not create an HNSW index automatically; provision that
index through your reviewed Oracle deployment process before production use.

To enable durable conversation memory, provision the table in
[`scripts/oracle-conversation-memory.sql`](../scripts/oracle-conversation-memory.sql), then set
`ORACLE_CONVERSATION_TABLE` to that table identifier.

## Start the services

Load `.env`, then run FastAPI:

```bash
uvicorn src.api.app:app --reload
```

Check process health:

```bash
curl http://127.0.0.1:8000/health
```

In a second terminal with the same environment, run Streamlit:

```bash
streamlit run src/ui/app.py
```

Before the first customer question, index your knowledge documents as described
in [ingestion](ingestion.md).
