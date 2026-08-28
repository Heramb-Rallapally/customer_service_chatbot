# Retrieval Integration Review

| Area | Status | Severity | Finding |
|------|--------|----------|---------|
| `search` interface | BLOCKER | BLOCKER | Actual signature is `search(self, query, filters: Optional[RetrievalFilters] = None, top_k=5)`, not the required keyword-only `query`, `filters: Mapping[str, str]`, `top_k`. A plain mapping will fail when the store calls `filters.as_metadata()`. |
| Retrieval score contract | BLOCKER | BLOCKER | OracleVS forwards raw backend scores and explicitly leaves score direction backend/version-dependent. Conversation thresholds cannot safely consume these values. |
| Score normalization | HIGH | HIGH | No uniform `[0,1]` normalization exists: cosine can be `[-1,1]`, dot product is unbounded, Euclidean is transformed to `(0,1]`, and OracleVS is unknown/raw. |
| OracleVS ranking | HIGH | HIGH | Results preserve backend order despite unknown score direction, so a consumer cannot rely on descending `result.score`. |
| OracleVS fallback filtering | HIGH | HIGH | When backend metadata filters are unsupported, it retrieves only `k` results then filters locally. This can return fewer than `top_k` matching documents even when sufficient matches exist. |
| Shared result model | PASS | LOW | `RetrievalResult` supplies the required `document_id`, `content`, `score`, and `metadata` fields with finite float validation. |
| Mock implementation | PASS | LOW | In-memory retrieval is deterministic, applies exact metadata filters, and orders its own scores descending. |
| Test coverage | MEDIUM | MEDIUM | Tests do not assert the required Member 3 signature, mapping input support, Oracle score direction/range, normalization, or fallback-filter refill behavior. Tests could not be run in this shell because `pytest` is not installed for available `python3`. |

## 1. BLOCKERS

- `RetrievalService.search` does not satisfy Member 3's required callable contract.
- There is no stable retrieval-score contract across the production OracleVS and in-memory adapters. This makes any ConversationEngine confidence/escalation threshold unsafe.

## 2. HIGH-RISK ISSUES

- OracleVS raw score ordering can be wrong for the declared `score` field because backend score direction is intentionally unspecified.
- OracleVS fallback filtering can under-return matching results.
- `DOT` and cosine scores are not universally bounded to `[0,1]`, so even the in-memory implementation is unsuitable for shared fixed thresholds.

## 3. CONTRACT COMPATIBILITY

**NO.** The implementation does not satisfy:

```python
def search(
    *,
    query: str,
    filters: Mapping[str, str],
    top_k: int,
) -> Sequence[RetrievalResult]
```

Required incompatibilities:

- `query` is positional-capable instead of keyword-only.
- `filters` is `Optional[RetrievalFilters]`, not `Mapping[str, str]`.
- `top_k` is optional/defaulted rather than required.
- Passing the required mapping type fails at runtime.

## 4. SCORE CONTRACT

- Raw score type/meaning: `float`; in-memory meaning varies by metric, while OracleVS passes through backend-defined raw scores.
- Higher or lower is better: Higher for in-memory; **backend-dependent/unknown** for OracleVS.
- Final score range: Not guaranteed. In-memory cosine `[-1,1]`, dot product unbounded, Euclidean-derived similarity `(0,1]`; OracleVS unknown.
- Is it normalized to [0,1]: **NO**.
- Is higher always better: **NO** across all implementations.
- Compatible with ConversationEngine thresholds: **NO**.

## 5. REQUIRED CHANGES

- Change the public `search` signature to the exact keyword-only Member 3 contract; convert/validate its mapping internally into `RetrievalFilters`.
- Define one documented score convention for every adapter: finite normalized `[0,1]`, with higher always better.
- Convert OracleVS scores based on the configured metric, sort by normalized score descending, and reject/handle unsupported score semantics explicitly.
- Over-fetch and locally filter when OracleVS lacks backend metadata-filter support.
- Add contract tests for the exact signature, mapping filters, normalized ordering, and fallback filtering.

## 6. FINAL VERDICT

READY AFTER REQUIRED FIXES
