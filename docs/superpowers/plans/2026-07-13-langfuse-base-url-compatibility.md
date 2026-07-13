# Langfuse Base URL Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tracing honor the standard `LANGFUSE_BASE_URL` variable, remove `LANGFUSE_HOST`, and preserve explicit OTLP endpoint precedence.

**Architecture:** Keep endpoint resolution inside the pure `TraceConfig.from_env` boundary. Resolve explicit OTLP configuration first; otherwise derive Langfuse OTLP HTTP configuration from the standard base URL and key pair, with Langfuse Cloud as the key-only default.

**Tech Stack:** Python 3.11+, dataclasses, pytest, OpenTelemetry OTLP/HTTP.

## Global Constraints

- `OTEL_EXPORTER_OTLP_ENDPOINT` has highest precedence.
- `LANGFUSE_BASE_URL` is the only supported Langfuse instance URL variable.
- `LANGFUSE_HOST` must not affect endpoint resolution.
- Never log credentials or authorization header values.
- The exporter appends `/v1/traces` exactly once.

---

### Task 1: Standard Langfuse Base URL Resolution

**Files:**
- Modify: `backend/extensions/trace/config.py`
- Create: `backend/tests/test_trace_config.py`

**Interfaces:**
- Consumes: `TraceConfig.from_env(env: Optional[Dict[str, str]]) -> TraceConfig`
- Produces: endpoint resolution using `LANGFUSE_BASE_URL` with explicit OTLP precedence.

- [ ] **Step 1: Write failing endpoint-resolution tests**

```python
from extensions.trace.config import TraceConfig


def test_langfuse_base_url_builds_self_hosted_otlp_endpoint():
    cfg = TraceConfig.from_env(
        {
            "LANGFUSE_BASE_URL": " https://langfuse.example.com/ ",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )
    assert cfg.endpoint == "https://langfuse.example.com/api/public/otel"
    assert cfg.enabled is True


def test_explicit_otlp_endpoint_wins_over_langfuse_base_url():
    cfg = TraceConfig.from_env(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.com",
            "LANGFUSE_BASE_URL": "https://langfuse.example.com",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )
    assert cfg.endpoint == "https://collector.example.com"


def test_langfuse_host_is_ignored():
    cfg = TraceConfig.from_env(
        {
            "LANGFUSE_HOST": "https://legacy.example.com",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )
    assert cfg.endpoint == "https://cloud.langfuse.com/api/public/otel"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_trace_config.py -q
```

Expected: the self-hosted test fails because the current implementation ignores `LANGFUSE_BASE_URL`.

- [ ] **Step 3: Implement minimal standard-variable resolution**

Change the Langfuse fallback in `TraceConfig.from_env` to:

```python
base_url = (
    (env.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com")
    .strip()
    .rstrip("/")
)
endpoint = f"{base_url}/api/public/otel"
```

Remove `LANGFUSE_HOST` from the module documentation. Preserve the existing branch condition so this logic runs only when no explicit OTLP endpoint exists and both Langfuse keys are present.

- [ ] **Step 4: Verify focused and startup regressions**

Run:

```bash
cd backend
uv run pytest tests/test_trace_config.py tests/test_lean_main_boot.py -q
uv run ruff check extensions/trace/config.py tests/test_trace_config.py
```

Expected: all tests and static checks pass.

- [ ] **Step 5: Commit the implementation**

```bash
git add backend/extensions/trace/config.py backend/tests/test_trace_config.py
git commit -m "fix(trace): honor standard Langfuse base URL"
```
