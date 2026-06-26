# Agent SOUL & Tools — Design

**Date:** 2026-06-26
**Status:** Approved by delegation (user asleep; explicit authority to design, self-review, and implement autonomously)
**Branch:** `personal/yfei/agent-core`
**Builds on:** the lean agent service (`backend/agent/`, `backend/app/`). The agent already runs a full tool loop (`agent/agent.py` consumes `ctx.tools.openai_schema()` for the LLM call and `ctx.tools.dispatch(tc)` to execute) — today the lean path simply passes a one-line system prompt and an empty `ToolBox([])`. This design fills those two slots with a configurable **SOUL** and a tight, extensible **tool registry**.

## Problem

The lean agent has no identity and no capabilities:

- `app/builder.py` sets `system_prompt = request.instructions or "You are a helpful assistant."` — no identity, no personality, no operating principles, no tool-use guidance.
- `build_context` passes `tools=ToolBox([])` — the agent can never call a tool, even though the loop that would execute them already exists.

We want two things, both **configurable** and both **extensible via MCP and skills**:

1. A professional, layered **system prompt** with a configurable **SOUL** — the part that defines *who the agent is and how it behaves* (identity, personality, principles, constraints), so a user can specify a custom agent without touching the engine prompt.
2. A small, high-quality **default tool set** ("贵在精" — refined, not sprawling), selectable per-agent, plus a registry that **MCP servers and skills** plug into uniformly.

## Research — what professional agents do

Surveying widely-deployed agents (Claude / Claude Code, Cursor, OpenAI's ChatGPT & Codex, Devin, Manus, Perplexity), their system prompts converge on a **layered** structure, and their toolsets converge on a **small universal core**:

**System-prompt layers (consistent across products):**
1. **Identity & role** — who the agent is, its name, its job. ("You are Claude Code, Anthropic's CLI…", "You are Cursor, an AI pair programmer…")
2. **Capabilities** — what it can do and the environment it operates in.
3. **Communication style** — tone, verbosity, formatting rules (Claude Code: terse, no preamble; ChatGPT: structured, friendly).
4. **Operating principles** — how it works: plan, act, verify; don't fabricate; prefer action over asking; cite sources.
5. **Tool-use protocol** — when and how to call tools, argument discipline, not narrating tool calls, grounding answers in tool results.
6. **Safety boundaries** — concise refusals / guardrails.
7. **Runtime context** — date/time, environment, user locale.

The reusable insight: layers **1, 3, 4, 6** are *persona* (vary per agent → the **SOUL**); layers **2, 5** are *engine* (stable, depend on wired capabilities → generated from the tool registry); layer **7** is *runtime* (already injected per-turn by `build_messages` as the `[System Time: …]` header).

**Default toolset (the universal core):** nearly every general agent ships **web search** + **web/page fetch (browse)**; coding agents add file read/write/edit + shell; data agents add code execution. For a general, sandbox-less API agent the defensible, safely-implementable, individually-testable core is:

- **`web_search`** — find current information (the single most impactful tool for a chat agent).
- **`web_fetch`** — read a specific URL's main text (pairs with search; also stands alone).
- **`current_datetime`** — trivial, zero-dependency, deterministic; resolves the most common "what's today / day-of-week" failure.

This trio is the "精" core. Everything else (code exec, KB retrieval, file ops, domain APIs) arrives via **skills** and **MCP**, not by bloating the default.

## Goals

- A pydantic **`Soul`** model capturing the persona layers, with a sensible professional **default soul**, mergeable per-request.
- **`render_system_prompt(soul, tool_names, …)`** — composes SOUL (persona) + a stable engine prompt (capabilities + tool protocol + safety) into the `system_prompt` string. Crisp and professional, not bloated.
- A **`ToolRegistry`** + the default tool trio, each its own module, individually enable-able; web tools degrade cleanly when unconfigured.
- **Extensibility:** a uniform path for **skills** (local, register `Tool`s) and **MCP** (adapter mapping an MCP tool schema → `Tool`), both feeding the same registry; SOUL's `tools_enabled` governs which an agent may use.
- Wire SOUL + registry into `build_context`/`AppState`; configurable via `Settings` and per-request.
- Fully TDD'd; the lean service stays import-lean and green.

## Scope

**In scope (this design → two plans):**
- **Plan A — SOUL & system prompt:** `agent/soul.py` (`Soul` + `DEFAULT_SOUL` + `render_system_prompt`), `build_context` composition (default soul ← request `soul` override ← `instructions` as `extra_instructions`), `ResponsesRequest.soul`, `AppState.soul`, settings hooks.
- **Plan B — Tools & extensibility:** `agent/tools/registry.py` (`ToolRegistry`), `agent/tools/builtin/` (`current_datetime`, `web_fetch`, `web_search` with a `SearchProvider` protocol), `agent/tools/skills.py` (local skill loader), `agent/tools/mcp.py` (MCP-schema→`Tool` adapter + provider interface), `build_default_registry(settings)`, `build_context` toolbox wiring, `AppState.registry`, settings hooks.

**Out of scope (designed, deferred to follow-ups — explicitly noted so they aren't mistaken for done):**
- A **live MCP transport** (stdio/SSE/websocket client). Plan B implements the *adapter* (MCP tool schema → `Tool`, calling an injected MCP client) and a provider interface, fully tested against a fake client; wiring a real transport + lifecycle is a follow-up.
- A **real search backend**. Plan B implements the `web_search` tool against a `SearchProvider` protocol with a configurable HTTP provider behind settings and a deterministic fake for tests; choosing/contracting a vendor (Tavily/Serp/Bing) is a config-time concern.
- Code-execution / file-system tools (need a sandbox; safety scope).
- Named multi-soul presets / a soul store + management UI (the model supports it; persistence/CRUD is later).
- Frontend surfacing of soul/tool selection.

## Architecture

### SOUL (`backend/agent/soul.py`)

```python
class Soul(BaseModel):
    name: str = "Aria"
    role: str = "a general-purpose AI assistant"
    identity: str = "<one paragraph: who the agent is>"
    personality: list[str] = [...]      # voice / traits
    principles: list[str] = [...]       # how it operates
    expertise: list[str] = []           # domains (optional)
    style: str = "<communication & formatting guidance>"
    constraints: list[str] = [...]      # guardrails / refusals
    extra_instructions: str = ""        # free-form (e.g. request.instructions)
    tools_enabled: Optional[list[str]] = None  # None = all registered defaults
```

`DEFAULT_SOUL` is a complete, professional general-assistant persona. `Soul.merge(override: dict) -> Soul` produces an effective soul (shallow field override; lists replaced, not concatenated, except `extra_instructions` which is set from `instructions`).

`render_system_prompt(soul: Soul, *, tool_names: list[str], extra: str = "") -> str` composes, in order:

1. **`# Identity`** — `You are {name}, {role}.` + `identity`; `Your expertise: …` if any.
2. **`# Personality`** — bulleted `personality`; `style`.
3. **`# Operating principles`** — bulleted `principles`.
4. **`# Tools`** (engine, stable) — a professional tool-use protocol (use a tool when it materially helps; pass well-formed arguments; never invent tool output; ground claims in results; take one logical action at a time; stop when the task is done) followed by the available `tool_names` (or "You have no tools enabled in this session." when empty).
5. **`# Safety`** (engine, stable) — concise boundaries (decline harmful/illegal requests; don't fabricate facts or citations; respect privacy).
6. **`extra_instructions`** (and any `extra`) appended under **`# Additional instructions`** when present.

The function is **pure** (no I/O, no clock) — the time header stays a per-turn concern of `build_messages`. This makes the whole persona layer unit-testable by string assertions.

### Tools

**`Tool`** (existing, `agent/tools/base.py`) — unchanged: `name, description, parameters (JSON Schema), fn (async ⇒ str), return_direct`. **`ToolBox`** (existing) — unchanged; the agent loop already uses it.

**`ToolRegistry`** (`agent/tools/registry.py`, new):
```python
class ToolRegistry:
    def register(self, tool: Tool) -> None        # idempotent by name (last wins, warns)
    def get(self, name: str) -> Optional[Tool]
    def names(self) -> list[str]
    def build_toolbox(self, names: Optional[list[str]] = None) -> ToolBox
        # names=None -> all registered; unknown names skipped with a warning
```

**Built-in tools** (`agent/tools/builtin/`), each a `make_*_tool(...) -> Tool` factory returning a `Tool` whose `fn` is an async closure:
- `current_datetime` — no args; returns the formatted local time (reuses `utils.time_utils`).
- `web_fetch(url: str)` — httpx GET (timeout, redirects, size cap), strip HTML to readable text, truncate (~8k chars), return text or a clear error string. The httpx client is injectable for tests.
- `web_search(query: str, num_results: int = 5)` — delegates to a `SearchProvider` (Protocol: `async def search(query, num_results) -> list[{title, url, snippet}]`); formats results as text. `build_default_registry` registers it only when a provider is configured; tests inject a fake provider.

**`build_default_registry(settings) -> ToolRegistry`** — registers `current_datetime` and `web_fetch` always; `web_search` only if a search provider is configured (`settings.search_provider`/key). Returns the populated registry.

### Extensibility

Both extension paths produce `Tool`s and `register` them — the registry is the single junction, and SOUL's `tools_enabled` + the registry uniformly govern availability.

- **Skills (`agent/tools/skills.py`)** — a *skill* is a local Python module under a skills dir exposing `get_tools() -> list[Tool]` (and, by convention, a `SKILL.md` describing it). `load_skills(path, registry)` imports each skill module and registers its tools. This is the simplest extension and is fully implemented + tested (a temp skill dir in tests).
- **MCP (`agent/tools/mcp.py`)** — `mcp_tool_to_tool(spec: dict, call: Callable) -> Tool` maps an MCP tool descriptor (`{name, description, inputSchema}`) to a `Tool` whose `fn` invokes `call(name, args)` and stringifies the MCP result. `register_mcp_tools(specs, call, registry, prefix="")` registers a server's tools (optionally namespaced by `prefix` to avoid collisions). The mapping + registration are pure and tested with a fake `call`. A live MCP client (transport, handshake, lifecycle) injects its `list_tools()`/`call_tool()` into these functions — that wiring is the deferred follow-up.

### Integration (`build_context`, `AppState`, `Settings`)

- `AppState` gains `soul: Soul` (the default) and `registry: ToolRegistry`, built in `lean_main.py` lifespan from `Settings`.
- `build_context(request, store, *, soul: Soul, registry: ToolRegistry)`:
  1. **Effective soul** = `soul.merge(request.soul or {})`; if `request.instructions`, set `extra_instructions = request.instructions`.
  2. **Tool names** = `effective_soul.tools_enabled` (or all registry names).
  3. **ToolBox** = `registry.build_toolbox(tool_names)`.
  4. **system_prompt** = `render_system_prompt(effective_soul, tool_names=toolbox tool names)`.
  5. Assemble `AgentContext` with that `system_prompt` and `tools`.
- `ResponsesRequest` gains `soul: Optional[dict] = None` (a partial override merged over the default). `instructions` keeps working — now as `extra_instructions` appended to the composed prompt (a deliberate semantics change from "instructions == whole system prompt").
- `Settings` gains: `agent_name`, `agent_role` (quick default-soul overrides), `search_provider` (`"none"` default), `search_api_key`, `search_endpoint`, `skills_dir` (optional).

### Data flow (one turn)

Request → `build_context`: merge default soul with `request.soul`/`instructions` → pick tools (soul.tools_enabled ∩ registry) → build ToolBox → render the layered system prompt naming those tools → `AgentContext`. The agent loop (unchanged) advertises `tools.openai_schema()` to the model and dispatches calls through `tools.dispatch()`. Web tools that aren't configured simply aren't in the registry, so the prompt never advertises a tool the agent can't run.

## Error handling

- Tool `fn`s return error strings (never raise to the loop); `ToolBox.dispatch` already wraps exceptions/retries. `web_fetch`/`web_search` catch network/timeout and return a concise `"<tool> failed: …"` so the model can recover.
- An unconfigured `web_search` is **not registered** (not a runtime error). If a soul's `tools_enabled` names a missing tool, `build_toolbox` skips it with a warning.
- A skill module that fails to import is logged and skipped — one bad skill never breaks boot.

## Testing

- **SOUL (pure):** `Soul` defaults + `merge` (override replaces, `instructions`→`extra_instructions`); `render_system_prompt` includes identity/name/role, personality, principles, safety, and the named tools; lists no tools cleanly when empty; `extra_instructions` appears when set and is absent when blank.
- **Builder:** `build_context` composes a prompt containing the soul's name + the enabled tool names; `request.soul` override wins; `instructions` appears as additional instructions; `tools_enabled` filters the ToolBox; the existing `test_build_context_*` is updated to assert the composed prompt *contains* the instruction (not equality).
- **Registry:** register/get/names; `build_toolbox(None)` = all; named subset; unknown name skipped.
- **Built-ins:** `current_datetime` returns a time string; `web_fetch` returns extracted text from a mocked httpx response and a clean error on failure; `web_search` formats results from a fake provider; `build_default_registry` omits `web_search` when unconfigured and includes it when configured.
- **Extensibility:** `load_skills` registers tools from a temp skill dir (and skips a broken module); `mcp_tool_to_tool` maps schema→`Tool` and the `fn` calls the injected client with parsed args; `register_mcp_tools` namespaces with a prefix.
- **Endpoint smoke:** a `TestClient` turn whose fake LLM emits a tool call to `current_datetime` exercises advertise→dispatch end-to-end and persists the `function_call`/`function_call_output` items.
- Import-lean + boot/isolation gates stay green; runs from `backend/` with `python -m pytest ../tests/app -q` and `../tests/agent` where applicable.

## Implementation sequencing

Two plans, built in order, executed subagent-driven:

1. **`2026-06-26-agent-soul.md`** — `Soul`/`DEFAULT_SOUL`/`render_system_prompt`, `ResponsesRequest.soul`, `AppState.soul`, `build_context` SOUL composition, settings, tests. (No tools yet → `tool_names=[]`; prompt says "no tools enabled".)
2. **`2026-06-26-agent-tools.md`** — `ToolRegistry`, built-in trio, `SearchProvider`, `build_default_registry`, skills loader, MCP adapter, `AppState.registry`, `build_context` toolbox wiring, settings, tests + the end-to-end tool-call smoke.

## Risks / open questions

- **Prompt quality is subjective.** The default SOUL aims at "professional general assistant"; it's data, not code, so it's trivially tunable later. Kept concise to limit token cost.
- **`instructions` semantics change** (whole-prompt → additional-instructions). One existing builder test is updated. Any external caller relying on `instructions` fully replacing the system prompt would see composed output instead; acceptable for this pre-release lean service and arguably more correct.
- **MCP/search are interfaces, not live integrations** this iteration — by design. The adapters are real and tested; the transports/vendors are the documented next step. Surfacing this clearly avoids "looks done but isn't".
- **`web_fetch` HTML→text** is a deliberately simple strip+truncate (no JS rendering, no readability heuristics). Adequate for v1; a real extractor (e.g. trafilatura) is a later upgrade.
- **Tool namespacing.** Skills/MCP can collide with built-in names; `register` warns on overwrite and MCP registration supports a `prefix`. A strict collision policy can come later.
