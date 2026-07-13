# Langfuse Base URL Compatibility Design

## Context

The tracing extension currently derives its OTLP endpoint from
`LANGFUSE_HOST`. Current Langfuse tooling and documentation use
`LANGFUSE_BASE_URL`, so a self-hosted deployment that sets the standard variable
silently falls back to `https://cloud.langfuse.com` and receives HTTP 401.

## Decision

`TraceConfig.from_env` will resolve the Langfuse instance root in this order:

1. `LANGFUSE_BASE_URL`, when non-empty.
2. `LANGFUSE_HOST`, for backward compatibility.
3. `https://cloud.langfuse.com`.

An explicit `OTEL_EXPORTER_OTLP_ENDPOINT` continues to take precedence over all
Langfuse convenience variables. The resolved root is normalized by trimming
whitespace and trailing slashes, then `/api/public/otel` is appended. Existing
public/secret-key authentication and HTTP/protobuf behavior remain unchanged.

## Error and Compatibility Boundaries

- Empty or whitespace-only variables are ignored.
- Existing deployments that only define `LANGFUSE_HOST` keep their behavior.
- When both names are defined, `LANGFUSE_BASE_URL` wins because it is the
  current standard name.
- No credential values or authorization headers are added to logs.
- Exporter endpoint construction remains responsible for appending
  `/v1/traces` exactly once.

## Verification

Unit tests will cover standard-variable resolution, precedence over the legacy
name, legacy fallback, and explicit OTLP precedence. Existing tracing-disabled
and application boot tests must remain green.
