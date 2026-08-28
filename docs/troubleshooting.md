# Troubleshooting

## `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root. If the shell still cannot resolve the
package, activate the virtual environment and set:

```bash
export PYTHONPATH="$PWD"
```

## Ollama is not running

Start the local server, then use the project health check:

```bash
ollama serve
python -m src.ollama.health
```

Verify `OLLAMA_BASE_URL` matches the running service. `/health` on FastAPI is
only a process-liveness endpoint, so it does not prove Ollama availability.

## Ollama model is missing

Install the configured model names:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

Then rerun `python -m src.ollama.health` after loading your environment.

## Oracle connection fails

Check that `ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, and `ORACLE_DB_DSN` are set
in the process that starts FastAPI or indexing. Confirm the DSN is reachable
from the current machine and the database user has the required table/index
privileges. Do not paste passwords or full connection details into tickets or
logs.

## Vector dimension mismatch

`nomic-embed-text` requires `EMBEDDING_DIMENSION=768` and a matching Oracle
`VECTOR(768, FLOAT32)` column. The application validates existing table DDL and
fails safely when dimensions differ. Use a new or deliberately reset vector
table and re-index the complete corpus; never mix model embeddings.

## Retrieval is empty or low confidence

Confirm that documents were indexed into the table named by `ORACLEVS_TABLE`,
the embedding model and dimension match the table, and the queried document has
compatible metadata. The engine deliberately avoids unsupported answers when
retrieval evidence is absent or weak. See [ingestion](ingestion.md) and
[retrieval](retrieval.md).

## FastAPI or Streamlit does not start

Activate `.venv`, load `.env` manually, and start each process from the
repository root:

```bash
set -a
source .env
set +a
uvicorn src.api.app:app --reload
```

In a separate shell:

```bash
set -a
source .env
set +a
streamlit run src/ui/app.py
```

Ensure `API_BASE_URL` points to the FastAPI server. Streamlit does not create
backend services itself.

## `.env` appears ignored

That is expected. `src.config` reads environment variables but does not load
dotenv files. Source `.env` before every terminal command or configure values
through your process manager, shell profile, CI secret store, or container
environment.
