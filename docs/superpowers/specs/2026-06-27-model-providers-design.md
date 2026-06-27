# Model Providers — Design

**Date:** 2026-06-27
**Status:** Approved by delegation (design agreed in conversation; user said "go ahead to implement")
**Branch:** `personal/yfei/agent-core`
**Builds on:** the lean agent service (`backend/app/`). Today `request.model` is **cosmetic** — `LeanLLM` fixes its model at construction (`llm.py:66` uses `self.model`) and `AppState.make_agent()` always uses the single `self.llm`, so every turn runs `settings.default_model` on one client regardless of the requested model.

## Problem

The service supports exactly one model/provider (`openai_base_url` + `openai_api_key` + `default_model`). We want to configure **multiple model providers**, **route each request to the right one**, expose the catalog for discovery, and **change the catalog without a redeploy** (edit a local YAML, reload).

## Decisions

1. **A model catalog over the OpenAI-compatible client.** Reuse `LeanLLM` (a thin `AsyncOpenAI` wrapper) as-is; nearly every provider is reachable OpenAI-compatibly (OpenAI, Azure, DeepSeek, Moonshot, Together, Groq, Qwen/dashscope compatible-mode, vLLM, Ollama, OpenRouter, Anthropic's OpenAI-compat endpoint). One client class, N configured `base_url`/key/model entries. No per-vendor SDK code.
2. **Local `models.yaml`, reloadable.** Catalog is non-secret (keys referenced by **env-var name**, resolved at client-build time) so it lives in version control / a ConfigMap. A `reload` re-reads it at runtime, swapping the model map atomically and preserving warm clients for unchanged entries.
3. **Capabilities drive behavior.** Each model carries `context_window`/`max_output_tokens` (→ the agent's budget, per-model), `supports_tools` (→ whether the ToolBox is advertised), `supports_reasoning` (→ `enable_thinking`).
4. **Backward compatible.** If no `models.yaml` exists, synthesize a one-model catalog from the legacy `Settings` (`openai_*`, `default_model`) so existing env-only deployments keep working. `AppState` keeps `llm`/`default_model` as a fallback so existing tests (which construct `AppState` without a router) are unchanged.
5. **Multi-instance:** config is **eventually consistent** — each instance owns its YAML (GitOps/ConfigMap) and reloads independently; brief disagreement during a rollout is acceptable for a catalog. (Runtime state sync — the in-memory `RunManager` for resume/cancel — is a *separate* problem, tracked for the Redis-Streams follow-up, not this design.)

## Architecture (`backend/app/providers.py`, new)

- **`ModelConfig`** (pydantic): `id, provider, base_url, api_key_env: Optional[str]=None, api_key: Optional[str]=None, context_window=128000, max_output_tokens=8000, supports_tools=True, supports_reasoning=False, temperature: Optional[float]=None`. Key resolution: `api_key` (direct, for the fallback/tests) else `os.environ[api_key_env]` else `""`.
- **`ModelCatalog`** (pydantic): `default_model: str`, `models: List[ModelConfig]`.
- **`load_catalog(path, settings) -> ModelCatalog`** — parse YAML if `path` exists; else synthesize `[ModelConfig(id=settings.default_model, provider="openai", base_url=settings.openai_base_url, api_key=settings.openai_api_key, ...)]`.
- **`ProviderRouter`**:
  - `__init__(catalog, path=None)` → `_apply(catalog)`.
  - `_apply(catalog)`: build `_configs: dict[id, ModelConfig]`, **skipping** a model whose `api_key_env` is set but resolves empty and has no direct `api_key` (log + omit — degrade cleanly, mirrors `web_search`); set `_default` (catalog default if present else first); **preserve** cached clients for unchanged configs (`new[id] == old[id]`), drop the rest.
  - `get_config(id) -> ModelConfig` (raises `KeyError` for unknown → route maps to 404).
  - `get_llm(id) -> LeanLLM` — build-and-cache: `LeanLLM(base_url, api_key=<resolved or "EMPTY">, model=id, max_tokens=cfg.max_output_tokens, enable_thinking=cfg.supports_reasoning, temperature=cfg.temperature or 0.7)`.
  - `register_llm(id, llm)` — inject a client (test seam + warm-override).
  - `default_model_id`, `list_models()`, `reload(catalog)`, `reload_from_disk()` (re-reads `_path` via `load_catalog`).
- **`AppState`** gains `router: Optional[ProviderRouter] = None` (keeps `llm`/`default_model` as fallback). `make_agent(llm=None, context_window=None, max_output_tokens=None)` lets the route pass a per-request client + per-model budget.
- **Route** (`create_response`): default `request.model` to `router.default_model_id` (or `state.default_model`); if router, `cfg = router.get_config(request.model)` (KeyError → 404 "unknown model"); gate tools via `registry=state.registry if (cfg is None or cfg.supports_tools) else None`; build the agent with `router.get_llm(request.model)` + `cfg.context_window`/`cfg.max_output_tokens`. No router → existing path unchanged.
- **Routes** (`backend/app/routes/models.py`, new): `GET /v1/models` (OpenAI-compatible list from the catalog) and `POST /v1/models/reload` (calls `router.reload_from_disk()`, returns the new list). Wired into `lean_main.py`; `lean_main` builds the router from `Settings.models_path`.
- **Settings:** `models_path: str = "models.yaml"`.

## Frontend

`newfrontend`: add `listModels()` (`GET /v1/models`); `ModelSelector` fetches the catalog and populates options instead of the hardcoded `["gpt-4o-mini","gpt-4o"]`, falling back to the chat store's current model if the fetch fails.

## Example `models.yaml`

```yaml
default_model: gpt-4o-mini
models:
  - id: gpt-4o-mini
    provider: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    context_window: 128000
    max_output_tokens: 16384
    supports_tools: true
  - id: claude-sonnet-4-6
    provider: anthropic
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    context_window: 200000
    supports_tools: true
    supports_reasoning: true
  - id: llama-3.3-70b
    provider: ollama
    base_url: http://localhost:11434/v1
    api_key_env: ""
```

## Testing

- **Router (unit):** load from a temp YAML; fallback synthesis when absent; `get_config`/`get_llm` build a `LeanLLM` with the right `base_url`/`model`/`enable_thinking`; unknown id raises `KeyError`; a model with a missing required key env is omitted; `reload` swaps configs and preserves a warm client for an unchanged model while dropping a changed one; key resolution precedence (direct `api_key` > env).
- **Route (TestClient):** a 2-model catalog with **injected fake clients** (`register_llm`) routes `model:"smart"` to the smart client and `model:"fast"` to the fast one (the echo text distinguishes); unknown model → 404; `supports_tools:false` → the system prompt advertises no tools; the no-router `AppState` path is unchanged (existing tests green).
- **Models endpoints:** `GET /v1/models` lists the catalog (OpenAI shape); `POST /v1/models/reload` re-reads a rewritten temp YAML and the new model appears.
- **Frontend:** `listModels()` parses `{data:[{id}]}`; `ModelSelector` renders fetched options and falls back on error. Existing tests stay green.
- Import-lean + boot/isolation gates green (`yaml` is already a dep).

## Out of scope / risks

- **Runtime-state sync across instances** (resume/cancel of a run on another instance) — separate problem; Redis-Streams `RunManager` is the tracked follow-up.
- **BYOK** (client-supplied provider keys) — keys stay server-side this iteration.
- **No auth on `/v1/models/reload`** — like the rest of the lean service; a real auth layer is deferred. Reload is idempotent and read-only w.r.t. data.
- **Native vendor features** (Anthropic prompt caching, native tool blocks) aren't exposed — the OpenAI-compat surface is the intentional tradeoff for leanness.
