# Langfuse Base URL Compatibility Design

## Context

The tracing extension currently derives its OTLP endpoint from the non-standard
`LANGFUSE_HOST`. Current Langfuse tooling and documentation use
`LANGFUSE_BASE_URL`, so a self-hosted deployment that sets the standard variable
silently falls back to `https://cloud.langfuse.com` and receives HTTP 401.

## Decision

`TraceConfig.from_env` will resolve the tracing endpoint in this order:

1. Explicit `OTEL_EXPORTER_OTLP_ENDPOINT`, when non-empty.
2. `LANGFUSE_BASE_URL`, when non-empty and both Langfuse keys are present.
3. `https://cloud.langfuse.com` when both Langfuse keys are present and no base
   URL is configured.

`LANGFUSE_HOST` is removed and ignored. The resolved Langfuse root is normalized
by trimming whitespace and trailing slashes, then `/api/public/otel` is
appended. Existing public/secret-key authentication and HTTP/protobuf behavior
remain unchanged.

## Error and Compatibility Boundaries

- Empty or whitespace-only variables are ignored.
- Deployments that only define `LANGFUSE_HOST` no longer enable tracing and must
  migrate to `LANGFUSE_BASE_URL`.
- `LANGFUSE_HOST` never affects endpoint resolution, including when both names
  are defined.
- No credential values or authorization headers are added to logs.
- Exporter endpoint construction remains responsible for appending
  `/v1/traces` exactly once.

## Verification

Unit tests will cover standard-variable resolution, removal of the legacy name,
and explicit OTLP precedence. Existing tracing-disabled and application boot
tests must remain green.
