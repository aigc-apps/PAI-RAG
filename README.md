<div align="center">

# PAI-RAG

### An agent core you can actually operate — a lean, OpenAI-Responses-compatible agent service with a web console.

</div>

PAI-RAG is a **self-hostable agent platform**: a server-side agent that runs a full
tool loop and speaks the OpenAI **Responses** wire format, paired with a React
console for authoring agents and operating the deployment. The backend carries
**zero heavy ML/RAG dependencies** (no `llama_index` / `torch` / `transformers`)
— retrieval is delegated to a pluggable vector engine (Elasticsearch), and code
execution to a sandbox provider.

## Repository layout

| Path | What it is | Toolchain |
| --- | --- | --- |
| [`backend/`](backend/) | Lean agent service — `app.lean_main:app` (FastAPI). Agent loop, model routing, knowledge bases, auth, skills, memory. | Python 3.11 · [uv](https://docs.astral.sh/uv/) |
| [`frontend/`](frontend/) | Web console — chat, Agent Studio (author), Control Room (operate). | React 19 · TypeScript · Vite |
| [`sandbox/`](sandbox/) | AgentRun sandbox image + NAS/skill mount runtime for `code_interpreter`. | Docker |
| [`docs/`](docs/) | Design docs, specs, and per-feature plans (`docs/superpowers/`, `docs/design/`, `docs/agent/`). | — |

## What's inside

- **SOUL / Org Persona** — a base persona (identity, principles, style, constraints)
  composed into a stable, cacheable system prompt; per-agent personas layer on top.
- **Model Connections** — a reloadable config routes each request to the right
  provider/model; credentials are referenced by **env-var name**, never stored inline.
- **Tools, Skills & MCP** — a tight default toolset behind a `ToolRegistry`, extensible
  with local skill packages and MCP adapters; enabled skills load into the per-turn context.
- **Knowledge bases** — per-agent knowledge scoping over an Elasticsearch vector engine,
  intersected with per-user permissions (a soft default that never widens access).
- **Auth (JWT + RBAC)** — first-run admin bootstrap, invite links, per-user data isolation.
- **Code execution** — `code_interpreter` backed by a sandbox provider (direct REST gateway
  or SDK adapter) with per-user/skill mounts.
- **Resilient streaming** — `background:true` detaches a run so it survives disconnects;
  resume and server-side cancel are supported.
- **User memory & rolling summary** — long-term per-user memory and a persisted conversation
  summary keep context small and prefix-cache-friendly.

## Quickstart (development)

Two dev servers — the Vite dev server proxies `/v1` to the backend on port 8000.

```bash
# 1) backend  (http://localhost:8000)
cd backend
uv sync                                                 # create .venv from uv.lock
cp data/config.yaml data/config.yaml.bak                # keep a copy before editing
export DASHSCOPE_API_KEY=sk-...                          # provider key(s) your config references
uv run uvicorn app.lean_main:app --reload --port 8000

# 2) frontend (http://localhost:5173)  — in a second terminal
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and complete first-run setup (create the admin account,
connect a model). See [`backend/README.md`](backend/README.md) for the full
configuration reference (`data/config.yaml`, env vars, sandbox, skills).

## Deployment (Docker)

A single combined image serves the whole app: nginx serves the built SPA and
reverse-proxies `/v1` to uvicorn in the same container.

```bash
docker build -t pai-rag .
docker run --rm -p 8080:80 \
  -e DASHSCOPE_API_KEY=sk-... \
  -v "$(pwd)/backend/data:/app/backend/data" \
  pai-rag
# open http://localhost:8080
```

See [`deploy/README.md`](deploy/README.md) for build args, runtime env, and the
CI/CD image push (Aliyun ACR).

## Testing

```bash
cd backend  && uv run pytest -q       # backend suite
cd frontend && npm test               # frontend (vitest)
```

## CI/CD

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — lint + test on push/PR
  (backend via `uv`, frontend via `npm`).
- [`.github/workflows/docker.yml`](.github/workflows/docker.yml) — build & push the
  combined image to Aliyun ACR on `feature`/`main` and `v*` tags.

## License

See [LICENSE](LICENSE).
