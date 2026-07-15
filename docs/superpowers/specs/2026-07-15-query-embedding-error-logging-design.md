# Query Embedding Error Logging Design

## Goal

When query embedding fails and knowledge search falls back to the search engine,
log enough model-service context to diagnose the failure without exposing the
configured API key.

## Scope

This change applies to the `knowledge.query_embedding` failure path. It does not
change embedding requests, retry behavior, fallback behavior, ingestion logging,
primary-search logging, or rerank logging.

## Log Contract

Every query-embedding failure warning includes:

- `operation=query_embedding`
- provider id
- qualified model id
- configured embedding dimension
- service URL
- exception type
- exception text
- original query text
- `fallback=search_engine`

For `httpx.HTTPStatusError`, it additionally includes the HTTP method, response
status code, request URL including its query parameters, and the full response
body. For connection, timeout, decoding, configuration, and incomplete-vector
errors, unavailable HTTP-only fields are omitted.

The service URL comes from the concrete embedder when available (`base_url` for
DashScope-native clients or `url` for OpenAI-compatible clients), then falls back
to the router's resolved model URL. A local embedder reports `local`.

## API-Key Redaction

The logger resolves the effective API key from the concrete embedder or model
configuration. Before emitting the warning, it replaces every literal occurrence
of a non-empty key in every logged string field with `***REDACTED***`.

Request headers are never logged. Query text, URL parameters, exception text,
and response bodies otherwise remain unredacted, as explicitly requested.

## Failure Behavior

Logging is best-effort and must not raise a second exception. Missing router
metadata, unreadable response bodies, or unusual exception objects fall back to
empty/unknown fields. Query embedding failure continues to set
`query_vector=None`, record the trace error, and execute the existing search
engine fallback.

## Implementation Boundary

A small helper in `backend/app/knowledge.py` constructs the structured context
at the catch point. Keeping it at this boundary provides access to the frozen KB
embedding config, query, embedder, router, and exception without changing all
retrieval clients or their public exception types.

## Tests

Tests verify:

- an HTTP 400 warning contains provider, model, dimension, method, request URL,
  service URL, status, exception text, response body, query, and fallback;
- API-key occurrences are redacted from every logged detail;
- a non-HTTP exception still logs its model/service context and error text;
- search results still use the existing fallback path;
- no Authorization header is logged.
