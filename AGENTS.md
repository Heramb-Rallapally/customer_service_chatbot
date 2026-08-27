# AGENTS.md

## 1. Project Overview

This repository contains a capstone project for a **Customer Service Chatbot with Multi-Turn Resolution**.

The system is a RAG-based customer support assistant that:

- Ingests product documentation, FAQs, historical support tickets, knowledge-base articles, troubleshooting guides, and video transcripts.
- Organizes knowledge using product, issue type, severity, resolution category, and version metadata.
- Retrieves relevant information using OracleVS and OCI Generative AI Embeddings.
- Generates grounded responses using OCI Generative AI Cohere Command R+.
- Maintains context across multi-turn conversations.
- Asks clarification questions when required information is missing.
- Detects sentiment and frustration.
- Recommends related knowledge and historical solutions.
- Detects when escalation is appropriate.
- Supports knowledge feedback and analytics.
- Exposes functionality through FastAPI.
- Provides a Streamlit user interface.

The project is developed by a 5-person team using Codex. **Integration safety and modularity are first-class requirements.**

---

## 2. Technology Stack

Use the following technologies unless the team explicitly agrees otherwise:

- Python 3.x
- Oracle AI Database 23ai
- OracleVS through LangChain
- OCI Generative AI Embeddings
- OCI Generative AI Cohere Command R+
- FastAPI
- Streamlit
- Pydantic
- Pandas
- Plotly
- pytest

Do not introduce another database or vector database without explicit team approval.

---

## 3. Repository Structure

Keep the project organized approximately as follows:

```text
customer-service-chatbot/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── conversation/
│   ├── proactive/
│   ├── knowledge/
│   ├── api/
│   ├── ui/
│   ├── db/
│   └── config.py
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── scripts/
├── examples/
├── docs/
├── requirements.txt
├── .env.example
├── README.md
└── AGENTS.md
```

Keep business logic inside `src/`. Do not place important application logic directly inside Streamlit pages or FastAPI route handlers.

---

## 4. Team Ownership

### Member 1 — Knowledge Ingestion + Database

Owns:

- `src/ingestion/`
- `src/db/`

Responsibilities:

- Source loading
- Document cleaning
- Chunking
- Metadata extraction
- Product categorization
- Issue type classification
- Severity tagging
- Resolution categorization
- Version mapping
- Oracle database persistence

### Member 2 — RAG / Retrieval

Owns:

- `src/retrieval/`

Responsibilities:

- OCI embeddings
- OracleVS
- Vector insertion
- Similarity search
- Metadata filtering
- Reranking where appropriate
- Retrieval evaluation
- Context preparation

### Member 3 — Conversation Engine

Owns:

- `src/conversation/`

Responsibilities:

- Conversation state
- Multi-turn memory
- Intent/context understanding
- Clarification questions
- RAG orchestration
- Resolution generation
- Confirmation and verification
- Resolution status
- Escalation decisions

### Member 4 — Proactive Support

Owns:

- `src/proactive/`

Responsibilities:

- Sentiment analysis
- Frustration detection
- Escalation signals
- Related article recommendations
- Similar issue matching
- Historical solution signals
- Customer history integration

### Member 5 — API + UI + Analytics / Integration

Owns:

- `src/api/`
- `src/ui/`
- Analytics/integration components

Responsibilities:

- FastAPI
- Streamlit
- Analytics dashboard
- End-to-end integration
- Integration testing
- Application startup/documentation

Member 5 acts as the **integration owner**, but architectural decisions remain team decisions.

---

## 5. Architecture Principles

The application should follow this general flow:

```text
User
  ↓
FastAPI / Streamlit
  ↓
Conversation Engine
  ├──→ Proactive Support
  │
  └──→ Retrieval
          ↓
       OracleVS
          ↓
   Knowledge Store
  ↓
LLM / Response Generation
  ↓
Resolution / Clarification / Escalation
```

### Layering rules

- UI must not directly access OracleVS.
- UI must not contain RAG or business logic.
- API routes should delegate to application services.
- Conversation logic should use the retrieval interface rather than implementing vector search itself.
- Proactive services should return structured signals rather than directly controlling the UI.
- Database access should be isolated behind database/repository components.
- External OCI services should be isolated behind reusable service interfaces.
- Modules must communicate through defined contracts.

---

## 6. Shared Data Contracts

Use shared Pydantic models or dataclasses for cross-module communication.

Do not create duplicate representations of the same object.

### KnowledgeDocument

Expected fields:

```python
id
content
source
product
issue_type
severity
resolution_category
version
metadata
```

### RetrievalResult

Expected fields:

```python
document_id
content
score
metadata
```

### ConversationState

Expected fields:

```python
conversation_id
user_id
messages
product
version
issue_type
issue_summary
severity
resolution_status
```

### ChatResponse

Expected fields:

```python
message
citations
suggested_actions
escalation_required
confidence
related_articles
```

If a contract must change, first inspect all consumers and update tests accordingly. Do not silently break an existing public interface.

---

## 7. Integration Rules

These rules are mandatory.

1. Read the existing code before making changes.
2. Read this `AGENTS.md` before implementing a feature.
3. Reuse existing models, services, and utilities.
4. Do not create duplicate utilities for existing functionality.
5. Do not redesign the entire application to solve a local problem.
6. Keep changes focused on the requested task.
7. Do not modify another team's module unnecessarily.
8. Preserve existing public interfaces unless a change is explicitly required.
9. Do not silently change database schemas.
10. Do not introduce unnecessary dependencies.
11. Update `requirements.txt` when a new dependency is approved.
12. Update `.env.example` when adding configuration.
13. Never hardcode credentials, API keys, OCIDs, passwords, or secrets.
14. Never commit `.env` or other secret files.
15. Use structured logging instead of scattered `print()` calls for application behavior.
16. Handle external-service failures gracefully.
17. Do not fabricate documents, citations, historical tickets, or retrieval results.
18. Do not claim that documentation supports an answer when no supporting knowledge was retrieved.
19. Prefer a small compatible change over a large refactor.
20. Do not delete working functionality without explicit approval.

---

## 8. RAG Behavior

The expected RAG flow is:

```text
User Message
    ↓
Conversation Context
    ↓
Intent / Context Analysis
    ↓
Retrieval
    ↓
Metadata Filtering
    ↓
Optional Reranking
    ↓
Context Construction
    ↓
Cohere Command R+
    ↓
Grounded Response
    ↓
Confidence / Resolution Decision
```

Knowledge priority should generally be:

1. Official product documentation
2. Troubleshooting guides
3. Knowledge-base articles
4. FAQs
5. Historical resolved tickets

Historical tickets provide useful evidence about previous resolutions but should not automatically override authoritative product documentation.

The chatbot should prefer information matching:

- product
- version
- issue type
- severity

when those attributes are known.

---

## 9. Grounding and Hallucination Control

The chatbot must prioritize correctness over producing an answer at all costs.

If relevant knowledge cannot be retrieved:

- Ask for clarification when additional information may help.
- Explain that sufficient information was not found when appropriate.
- Recommend escalation when the issue cannot be safely resolved.
- Never invent a troubleshooting step and present it as official documentation.

Responses should include source references/citations whenever the interface supports them.

---

## 10. Multi-Turn Conversation Requirements

The conversation engine must retain information already provided by the customer.

Example:

```text
User:
My VPN isn't working.

Assistant:
Which VPN client are you using?

User:
Oracle VPN.

Assistant:
What version are you using?

User:
5.2.
```

The system should retain:

```text
product = Oracle VPN
version = 5.2
```

The chatbot must not repeatedly ask for information that is already known.

The conversation engine should support:

- clarification
- context retention
- follow-up questions
- resolution confirmation
- retry after failed troubleshooting
- escalation after repeated failure

---

## 11. Escalation Rules

Escalation may be triggered by:

- Low retrieval confidence
- Unsupported issue
- High severity
- Repeated unsuccessful troubleshooting
- Strong customer frustration
- Explicit human-support request
- Other clearly defined business rules

Escalation should be represented as structured state/data, not merely a sentence in the generated response.

Example:

```python
{
    "escalation_required": True,
    "reason": "repeated_failed_troubleshooting",
    "severity": "high"
}
```

---

## 12. Proactive Support

Proactive features must influence the support workflow where appropriate.

For example:

```text
Negative sentiment
      ↓
Frustration increases
      ↓
Avoid repeating failed steps
      ↓
Consider alternative historical resolution
      ↓
Escalate if necessary
```

Do not implement proactive features only as dashboard statistics. They should have useful effects on the customer-support experience.

---

## 13. Historical Support Tickets

Historical tickets should have rich metadata where available:

```text
source
product
version
issue_type
severity
resolution
resolution_time
ticket_status
```

The system should be capable of finding similar historical issues and extracting useful resolution patterns.

Historical tickets must be treated as supporting evidence and not automatically as authoritative product documentation.

---

## 14. Configuration and Secrets

All environment-specific configuration must come from environment variables or a centralized configuration layer.

Example:

```text
OCI_CONFIG_PROFILE
OCI_COMPARTMENT_ID
OCI_ENDPOINT
EMBEDDING_MODEL
LLM_MODEL
ORACLE_DB_USER
ORACLE_DB_PASSWORD
ORACLE_DB_DSN
```

Use `.env.example` to document required variables.

Never commit:

- API keys
- OCI private keys
- passwords
- tokens
- `.env`
- production credentials

---

## 15. Testing Requirements

Every feature must include appropriate tests.

At minimum, cover:

- Normal behavior
- Invalid input
- Missing information
- Empty retrieval results
- External service failures
- Multi-turn context retention
- Escalation
- Low-confidence retrieval
- Error handling

Use mocks for OCI and other external services in unit tests.

Keep real-service tests separate from unit tests and make them explicitly identifiable.

Suggested structure:

```text
tests/
├── unit/
├── integration/
└── evaluation/
```

Run relevant tests before committing.

---

## 16. Evaluation

Create a small evaluation dataset containing realistic customer-support scenarios.

Evaluate at least:

### Retrieval

- Recall@K
- MRR where appropriate

### Resolution

- Correctness
- Completeness
- Groundedness

### Multi-Turn

- Context retention
- Clarification quality
- Resolution success

### Proactive Support

- Sentiment accuracy
- Frustration detection
- Escalation accuracy

Evaluation should be repeatable and stored in a documented location.

---

## 17. Error Handling

External services can fail.

The application should gracefully handle:

- OCI authentication errors
- Database connection failures
- OracleVS errors
- Embedding failures
- LLM failures
- Empty retrieval results
- Invalid user input
- Timeouts

Do not expose secrets or sensitive infrastructure details in user-facing errors.

Use clear internal logs for debugging.

---

## 18. API Rules

FastAPI routes should be thin.

Preferred pattern:

```text
API Route
   ↓
Application Service
   ↓
Domain / Business Logic
   ↓
Repository / External Service
```

Do not place complex RAG orchestration directly inside route functions.

Use Pydantic request/response schemas.

Document API endpoints through FastAPI/OpenAPI.

---

## 19. UI Rules

Streamlit should be responsible for presentation and user interaction.

Do not put:

- vector search implementation
- database queries
- complex prompt construction
- business rules

directly in UI code.

The UI should consume API/application services.

The chatbot should clearly display, where available:

- Assistant response
- Relevant sources
- Suggested actions
- Resolution status
- Escalation status

---

## 20. Analytics Requirements

The analytics layer should support metrics such as:

- Resolution rate
- Average response time
- Resolution time
- Customer satisfaction score
- Escalation rate
- Most common issues
- Sentiment distribution
- Frustration trends
- Knowledge gaps
- Popular queries

Analytics code should not affect the core chatbot path unnecessarily.

---

## 21. Knowledge Feedback Loop

Support feedback should help identify:

- Missing documentation
- Frequently asked questions
- Poorly resolved issues
- Popular queries
- Knowledge gaps
- Articles requiring updates
- New solutions worth adding

Do not automatically modify authoritative knowledge without an explicit validation/review process.

---

## 22. Git Workflow

Use feature branches.

Recommended structure:

```text
main
└── develop
    ├── feature/ingestion
    ├── feature/retrieval
    ├── feature/conversation
    ├── feature/proactive
    └── feature/api-ui
```

Do not commit directly to `main`.

Before opening a PR:

1. Run tests.
2. Review `git diff`.
3. Check for accidental files/secrets.
4. Confirm public interfaces remain compatible.
5. Update documentation if needed.
6. Clearly describe integration impact.

---

## 23. Codex Development Workflow

Before modifying code:

### Step 1 — Inspect

Inspect:

- repository structure
- relevant source files
- existing interfaces
- tests
- configuration
- dependencies

### Step 2 — Plan

Briefly identify:

- files to change
- interfaces involved
- dependencies
- tests required
- integration risks

### Step 3 — Implement

Make the smallest clean implementation that satisfies the task.

### Step 4 — Test

Run relevant tests.

If a test fails:

- determine whether the failure is caused by the change
- fix the underlying issue
- do not simply remove or weaken the test

### Step 5 — Review

Review the final diff for:

- unnecessary changes
- duplicated code
- secrets
- broken interfaces
- unrelated formatting changes
- missing tests

### Step 6 — Report

When finished, report:

- What was changed
- Tests run
- Test results
- Any limitations
- Any integration considerations

Do not claim a feature is complete if it has not been tested.

---

## 24. Mock-First Development

To allow all five team members to work in parallel, modules should initially support mocked dependencies.

Examples:

```text
Conversation Engine
        ↓
Mock Retriever
```

and:

```text
API/UI
   ↓
Mock Conversation Service
```

This allows development before all real OCI components are available.

Mocks should implement the same interfaces as the real services.

---

## 25. Dependency Rules

Before adding a dependency:

1. Check whether the existing project already provides the functionality.
2. Check whether the Python standard library is sufficient.
3. Check whether an existing dependency can provide it.
4. Add a new dependency only when justified.

Avoid dependency duplication and unnecessary frameworks.

---

## 26. Database Rules

Database schema changes must be deliberate.

Before changing a schema:

- inspect existing models/schema
- identify all consumers
- document the change
- update relevant tests
- update initialization/setup scripts

Do not create multiple competing tables for the same conceptual entity.

---

## 27. Documentation Rules

Update documentation when behavior or setup changes.

Important documentation includes:

- README.md
- API documentation
- Setup instructions
- Environment variables
- Architecture documentation
- Data format documentation
- Evaluation instructions

A new developer should be able to clone the repository and understand how to run the project from the README.

---

## 28. Definition of Done

A feature is considered complete only when:

- [ ] Implementation is complete.
- [ ] Existing architecture is respected.
- [ ] Shared interfaces are preserved.
- [ ] Appropriate tests exist.
- [ ] Tests pass.
- [ ] Error handling exists.
- [ ] No credentials/secrets are committed.
- [ ] Configuration is documented.
- [ ] Integration impact has been considered.
- [ ] Documentation is updated where necessary.
- [ ] The implementation does not unnecessarily duplicate existing functionality.

---

## 29. Final Principle

This is a shared team repository.

**Do not optimize only for making the requested feature work in isolation. Optimize for making the feature work safely with everything else.**

When choosing between:

```text
Large redesign
vs.
Small compatible change
```

prefer the **small compatible change** unless there is a clear architectural reason to redesign.

When uncertain:

1. Inspect existing code.
2. Preserve existing contracts.
3. Reuse existing functionality.
4. Keep changes modular.
5. Add tests.
6. Ask for clarification rather than silently changing architecture.

**Integration safety is more important than local elegance.**
