# backend — Lean Agent Service (the base)

This is the **clean base** for PAI-Loop's agent core: a lean, OpenAI-Responses-compatible
agent service with **zero heavy ML/RAG dependencies** (no `llama_index`, `torch`,
`transformers`, `chromadb`, …). It was extracted from the legacy pairag backend
(see [docs/MIGRATION.md](docs/MIGRATION.md) for the history).

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)**.

## Quickstart

```bash
cd backend
uv sync                      # create .venv + install from pyproject/uv.lock
uv run pytest -q             # run the test suite (currently 165 tests)
uv run uvicorn app.lean_main:app --reload --port 8000   # run the service
```

Configure the unified agent config:

```bash
vim data/config.yaml                  # edit models, agents, tools, skills, providers
export OPENAI_API_KEY=sk-...          # or whichever provider key your config references
```

Enable cloud sandbox execution by filling `providers[].id=sandbox.default` in
`data/config.yaml`. The backend talks to the configured sandbox provider through
a small async provider abstraction, so production deployments can use a direct
REST gateway or an optional SDK-backed adapter without changing the
`code_interpreter` tool contract.

```bash
export AGENTRUN_SANDBOX_API_KEY=...   # optional gateway auth token
```

`sandbox.default.settings` supports a `templates` map of AgentRun sandbox images keyed
by template key (each key an object with `name`, and optionally `code_writable` and
`env_refs`), plus a `default_template` key used when an agent doesn't pick one. Each
agent binds to a specific template via `AgentProfile.sandbox.template` (a key into
`templates`; blank falls back to `default_template`). `sandbox.default.settings` also
carries user/tenant/conversation isolation, idle timeout, execution timeout up to 30
seconds, cwd, OSS mounts for custom skills, and NAS mounts for user files; the agent
sees this as the `code_interpreter` tool once the sandbox capability is enabled.

The default REST gateway contract follows the AgentRun sandbox shape:

- `POST /sandboxes` creates or returns a scoped sandbox instance.
- `POST /sandboxes/{sandbox_id}/contexts/execute` runs code in a sandbox context.
- `POST /sandboxes/{sandbox_id}/stop` releases an idle sandbox instance.

Enable custom skills by placing skill packages under the configured skill root:

```text
data/skills/report-writer/
  skill.yaml
  SKILL.md
  resources/
  scripts/
```

```yaml
skills:
  root: ./data/skills
  mount:
    mount_root: /mnt/skills
```

Each package's `skill.yaml` declares metadata, triggers, and required tools;
`SKILL.md` contains the model-facing workflow. Matching enabled skills are
loaded into the per-turn context as `# Active Skills`. The active agent's
enabled skills also determine the read-only skill mounts attached to its
sandbox scope.

## What it does

An agent that runs a full server-side tool loop and serializes to the OpenAI **Responses**
wire format (`/v1/responses`, streaming via `reasoning.summary` + `output_text` events).
Features already built (each has a design + plan under `../docs/superpowers/`):

- **Configurable SOUL** — persona (identity/personality/principles/style/constraints) composed
  into a **stable, cacheable system prompt**; `+ project context` layer.
- **Tools** — a tight default set (`current_datetime`, `web_fetch`, `web_search`) behind a
  `ToolRegistry`, extensible via **local skills** and **MCP** adapters.
- **Model providers** — a reloadable `data/config.yaml` `models:` section routes each request to the right
  provider/client with per-model capabilities (`GET /v1/models`, `POST /v1/models/reload`).
- **Resilient streaming** — `background:true` detaches the run so it survives disconnects;
  **resume** (`GET /v1/responses/{id}?stream=true&starting_after=N`) and server-side
  **cancel** (`POST /v1/responses/{id}/cancel`).
- **User memory** — per-user long-term memory (OpenAI `user`/`safety_identifier` aligned),
  injected into context, written by a background extraction/consolidation pipeline;
  managed via `/v1/users/{id}/memories`.
- **Context construction v2** — a **rolling, persisted conversation summary** bridges old
  turns; the prompt is layered as `[stable system] + history + [volatile context] + user turn`
  for prefix caching.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /v1/responses` | Run a turn (sync or `stream`); `background:true` for resumable runs |
| `GET /v1/responses/{id}` | Fetch a stored response; `?stream=true&starting_after=N` to resume |
| `POST /v1/responses/{id}/cancel` | Server-side cancel of an in-flight run |
| `DELETE /v1/responses/{id}` | Delete a stored response |
| `GET /v1/conversations[?user_id=]` · `GET /v1/conversations/{id}` · `DELETE …` | Conversation list/detail/delete |
| `GET /v1/models` · `POST /v1/models/reload` | Model catalog list + hot reload |
| `GET/DELETE /v1/users/{id}/memories[/{memory_id}]` | User-memory management |
| `POST /v1/chat/completions` | Legacy chat-completions shim |

## Structure (flat layout; packages at the project root)

```
backend/
  pyproject.toml          # uv project: deps + pytest config (pythonpath=["."])
  uv.lock                 # pinned, reproducible env (no heavy ML deps)
  data/config.yaml        # unified config: models, agents, providers, capabilities
  app/                    # the service
    lean_main.py          # FastAPI app + lifespan (builds AppState: store, llm, soul, registry, router, …)
    config.py             # Settings (env-driven)
    deps.py               # AppState + make_agent + get_state
    schemas.py            # ResponsesRequest
    builder.py            # build_context: assembles AgentContext (stable prompt + volatile block + history)
    llm.py                # LeanLLM (streaming wrapper over AsyncOpenAI)
    providers.py          # ProviderRouter + model catalog (config.yaml models section)
    memory.py             # user-memory extraction/consolidation pipeline
    summarizer.py         # rolling conversation summary pipeline
    runs.py               # RunManager (detached background runs, resume, cancel)
    conversations_view.py # item-log → UI message grouping
    db.py · models.py     # SQLModel engine + tables
    store/                # ResponseStore: base (Protocol) · memory (InMemoryStore) · sql (SqlStore)
    routes/               # responses · conversations · models · users · chat
  agent/                  # provider-agnostic agent core
    agent.py              # Agent.run loop (build_messages, tool loop, AgentEvents)
    context.py message.py budgeting.py   # AgentContext, Message, AgentMessageManager (context budget)
    soul.py               # Soul + render_stable_system_prompt / render_context_block
    core/events.py        # AgentEvent types
    tools/                # base (Tool/ToolBox) · registry · builtin/ · skills · mcp · adapter
  api/protocol/           # responses_serializer (AgentEvent → OpenAI Responses) · chat_serializer
  common/llm/models.py    # LLM chunk contract (TextChunk/ReasoningChunk/ErrorChunk)
  memory/utils.py         # token estimate/truncate helpers (tokenizer-optional)
  utils/                  # time/json/constants helpers
  tests/                  # the lean test suite (flat; pythonpath=["."])
  docs/MIGRATION.md       # how the rest of backend/ migrates here
```

## Configuration (env vars / `Settings`)

| Var | Default | Purpose |
|---|---|---|
| `CONFIG_PATH` / `MODELS_PATH` | `./data/config.yaml` / `./data/config.yaml` | Unified config file; both point to the same YAML during development |
| `DB_URL` / `STORE_BACKEND` | `sqlite+aiosqlite:///./data/agent.db` / `sql` | Store backend (`sql` \| `memory`) |
| `AGENT_NAME` / `AGENT_ROLE` | `Aria` / general assistant | Default SOUL identity |
| `PROJECT_CONTEXT` | "" | Static project context injected into the stable system prompt |
| `MEMORY_ENABLED` / `MEMORY_MODEL` / `MEMORY_INJECT_LIMIT` | `false` / "" / `30` | User-memory extraction + injection |
| `SUMMARY_ENABLED` / `SUMMARY_KEEP_RECENT` / `SUMMARY_BATCH` | `false` / `20` / `20` | Rolling conversation summary |
| `SEARCH_PROVIDER` / `SEARCH_API_KEY` / `SEARCH_ENDPOINT` | `none` / "" / "" | `web_search` provider (off until configured) |
| `SKILLS_DIR` | "" | Directory of local skill modules (`get_tools()`) |

## Dependency management (uv)

- `pyproject.toml` declares runtime deps + a `dev` group (`pytest`); `uv.lock` pins everything.
- `uv sync` creates `.venv` and installs the locked set; `uv add <pkg>` / `uv remove <pkg>` to change deps.
- `uv run <cmd>` runs inside the env. The lock is intentionally **heavy-dep-free**; the
  `tests/test_lean_import_isolation.py` test enforces that the service imports with the ML/trace
  stack blocked.

## Notes / known follow-ups

Some agent modules still carry **guarded** legacy hooks (e.g. tracing via `extensions.trace`,
the llama-index `tools/adapter.py`) that import lazily and are inert here — they'll be cleaned
during migration. Background pipelines (memory + summary) use fire-and-forget `asyncio.create_task`;
hardening (a tracked task set) is a tracked follow-up. See `../docs/superpowers/` for per-feature
designs and deferred items.
