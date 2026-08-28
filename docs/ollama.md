# Ollama

Ollama is the default local provider for both generation and embeddings. The
application talks to its local HTTP API through `src.ollama.OllamaApiClient`.

## Models

| Role | Default model | Requirement |
| --- | --- | --- |
| LLM | `llama3.2:3b` | Produces structured grounded responses. |
| Embeddings | `nomic-embed-text` | Produces 768-dimensional vectors. |

Install and verify them:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

Start a server if your installation does not do so automatically:

```bash
ollama serve
```

After loading your environment, run the project health check:

```bash
python -m src.ollama.health
```

The health command checks the configured base URL and required models. FastAPI
`GET /health` is intentionally different: it checks only whether the API
process is alive and does not contact Ollama or Oracle.

## Local configuration

```dotenv
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=120
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

The application validates model availability during lazy application
construction and validates every embedding response for the configured 768
dimensions.

## Switching to OCI

OCI adapters are retained for future use. Set `LLM_PROVIDER=oci` and/or
`EMBEDDING_PROVIDER=oci`, configure the selected model IDs and embedding
dimension, and provide the OCI configuration described in
[configuration](configuration.md). OracleVS remains the vector store in either
case. Rebuild the vector table and corpus if the embedding model changes.
