# Query Embedding Error Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit actionable query-embedding failure logs containing model-service and HTTP response details while redacting only the effective API key.

**Architecture:** Build a structured diagnostic dictionary at the existing `KnowledgeService.search` catch boundary, where the frozen KB configuration, concrete embedder, router, query, and exception coexist. Format one stable warning from that dictionary and preserve the existing fallback behavior.

**Tech Stack:** Python 3.12, Loguru, httpx, pytest.

## Global Constraints

- Query text, URL parameters, exception text, and response body remain visible.
- Every literal occurrence of the effective API key is replaced with `***REDACTED***`.
- Request headers are never logged.
- HTTP response bodies are logged in full.
- Logging failures cannot replace or interrupt the existing search-engine fallback.
- No embedding request, retry, ingestion, primary-search, or rerank behavior changes.

---

### Task 1: Structured Query-Embedding Failure Diagnostics

**Files:**
- Modify: `backend/app/knowledge.py:1068-1101`
- Test: `backend/tests/test_knowledge_retrieval_models.py:678-715`

**Interfaces:**
- Consumes: group `provider`, qualified `model`, `dimension`, query text, concrete embedder, `ProviderRouter`, and caught `Exception`.
- Produces: `_query_embedding_error_context(...) -> dict[str, object]` containing stable log fields with API-key redaction already applied.
- Preserves: `query_vector=None` and `fallback=search_engine` after every caught exception.

- [ ] **Step 1: Write failing HTTP-detail and API-key-redaction tests**

Use an `httpx.HTTPStatusError` whose request URL, exception message, and response
body contain a sentinel API key. Inject an embedder exposing `model`, `url`, and
`api_key`, then assert the captured warning contains:

```python
assert "provider=vendor" in message
assert "model=vendor/embed-v1" in message
assert "dimension=8" in message
assert "service_url=https://embed.example/v1/embeddings?region=cn" in message
assert "http_method=POST" in message
assert "http_status=400" in message
assert 'response_body={"error":"invalid model"}' in message
assert "query=diagnose this" in message
assert "fallback=search_engine" in message
assert api_key not in message
assert "***REDACTED***" in message
assert "Authorization" not in message
```

Add a plain `RuntimeError` case asserting provider/model/service URL/error text
are present and fallback results remain available.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd backend
.venv/bin/pytest -q tests/test_knowledge_retrieval_models.py \
  -k 'query_embedding_fallback'
```

Expected: the new assertions fail because the current warning contains only
`operation`, `error_type`, and `fallback`.

- [ ] **Step 3: Implement diagnostic extraction and redaction**

In `backend/app/knowledge.py`, import `httpx` and add focused helpers equivalent
to:

```python
def _redact_api_key(value: object, api_key: str) -> str:
    text = str(value)
    return text.replace(api_key, "***REDACTED***") if api_key else text


def _query_embedding_error_context(
    *, provider, model, dimension, query, embedder, router, error
):
    api_key = str(getattr(embedder, "api_key", "") or "")
    service_url = str(
        getattr(embedder, "url", "")
        or getattr(embedder, "base_url", "")
        or "unknown"
    )
    context = {
        "provider": provider,
        "model": model,
        "dimension": dimension,
        "service_url": service_url,
        "error_type": type(error).__name__,
        "error": str(error),
        "query": query,
    }
    if isinstance(error, httpx.HTTPStatusError):
        context.update(
            {
                "http_method": error.request.method,
                "request_url": str(error.request.url),
                "http_status": error.response.status_code,
                "response_body": error.response.text,
            }
        )
    return {
        key: _redact_api_key(value, api_key) for key, value in context.items()
    }
```

Make metadata lookup best-effort. When embedder construction fails, resolve URL
and key from `router.get_config(model)` without allowing lookup errors to escape.
Format the warning as stable `key=value` pairs in the field order defined by the
spec, ending with `fallback=search_engine`. Do not log headers.

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
cd backend
.venv/bin/pytest -q \
  tests/test_knowledge_retrieval_models.py \
  tests/test_retrieval_models.py \
  tests/test_providers.py
```

Expected: all tests pass; the existing fallback tests continue to return search
results.

- [ ] **Step 5: Commit and push**

```bash
git add backend/app/knowledge.py backend/tests/test_knowledge_retrieval_models.py
git commit -m "feat(knowledge): log query embedding failure details"
git push origin personal/yfei/agent-core
```
