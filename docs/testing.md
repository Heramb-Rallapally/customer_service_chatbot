# Testing and validation

Run commands from the repository root with the virtual environment activated.

## Credential-free suite

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/customer_service_chatbot_compile_cache \
  .venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
git diff --check
```

The unit suite uses injected fakes for Oracle, Ollama, retrieval, memory,
proactive services, and analytics. It covers shared contracts, conversation
behavior, API/UI boundaries, ingestion/indexing delegation, retrieval score and
filter logic, and evaluation utilities.

## Live OracleVS test

The real OracleVS test is skipped unless explicitly enabled. Use an isolated
test table, never a production knowledge table:

```bash
set -a
source .env
set +a

RUN_ORACLEVS_INTEGRATION=1 \
ORACLEVS_INTEGRATION_TABLE=<isolated_table> \
python -m pytest tests/integration/test_oraclevs_live.py
```

It also requires `ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, and `ORACLE_DB_DSN`.

## Live Ollama + Oracle test

This test performs real Ollama embeddings, OracleVS insertion/search, and local
Ollama generation. It also uses an isolated caller-provided table:

```bash
set -a
source .env
set +a

RUN_OLLAMA_ORACLE_INTEGRATION=1 \
ORACLEVS_INTEGRATION_TABLE=<isolated_table> \
python -m pytest tests/integration/test_ollama_oraclevs_live.py
```

Do not claim live compatibility from mocked tests alone. The live tests clean
up their own probe rows, but their tables and indexes are operator-provisioned.

## Evaluation

The evaluation runner exercises the existing application boundary and writes
only outcome metadata, not raw customer text or generated answers:

```bash
python -m src.evaluation.run \
  --dataset examples/evaluation/cases.json \
  --output examples/evaluation/results.json
```

This command uses the configured live composition root. For credential-free
evaluation examples and metrics, see [evaluation](evaluation.md).
