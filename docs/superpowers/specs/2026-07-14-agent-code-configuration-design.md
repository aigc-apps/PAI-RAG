# Agent Code Configuration Design

## Goal

Make code-repository access an explicit, per-Agent capability instead of a
sandbox-provider deployment flag. Disabled Agents must not receive code-layer
instructions. Enabled Agents use `AGENT_CODE_PATH` when it is set and otherwise
use `/opt/code`.

## Configuration model

Add an `AgentCodeConfig` model and store it on each `AgentProfile`:

```json
{
  "code": {
    "enabled": false,
    "manifest": ""
  }
}
```

`enabled` defaults to `false`. `manifest` is the optional, administrator-edited
Markdown description of available repositories. The existing top-level
`code_manifest` field is removed. Because this feature has not shipped, no legacy
fallback, migration, or dual-write behavior is required.

The complete Agent configuration document is already persisted and versioned by
`SqlAgentConfigStore`, so the nested object is stored in the existing database
document. No relational schema change or Alembic migration is needed.

## Runtime behavior

The builder reads code settings only from the resolved Agent profile. It injects
the code-layer prompt when both conditions hold:

1. `agent.code.enabled` is true.
2. The Agent's effective toolbox includes `shell` or `code_interpreter` through
   the sandbox capability.

The code prompt contains the Agent's `code.manifest`, when present, and instructs
the model to resolve the repository root with shell semantics equivalent to:

```sh
CODE_PATH="${AGENT_CODE_PATH:-/opt/code}"
```

The common shell tool description remains deployment-neutral and does not expose
the code layer to disabled Agents. The sandbox provider no longer owns or reads a
`code_layer_enabled` setting, and the environment-contract builder no longer
injects `AGENT_CODE_PATH`. A sandbox image may set it to override the default;
otherwise the Agent uses `/opt/code`.

Main Agents and subagents use the same resolved Agent code configuration. The
configuration affects prompt/tool guidance only; it does not grant filesystem
permissions beyond those already provided by the sandbox runtime.

## API and UI

The code-manifest generation endpoint resolves the requested Agent and requires:

- an existing Agent;
- `agent.code.enabled` to be true;
- an available sandbox provider and model router.

A disabled Agent receives HTTP 409 with a clear message. Missing Agents remain
404, and missing runtime dependencies remain 503. If the resolved code directory
is empty or unreadable, the generation Agent reports that condition in the
returned manifest instead of the API pretending that the layer is available.

The Agent settings UI exposes a manual code-access toggle. New Agents default to
off. The repository manifest editor and AI-generate action are shown only while
code access is enabled. Saving updates the nested `code.enabled` and
`code.manifest` fields through the existing whole-document configuration API.

## Removed behavior

- Remove `code_layer_enabled` from the sandbox provider model, default document,
  local configuration, prompt builder inputs, and shell-description gating.
- Remove the top-level Agent `code_manifest` field.
- Do not infer code access from a non-empty manifest, selected tools, sandbox
  template name, or directory presence.
- Do not inject `AGENT_CODE_PATH` from the backend.

## Validation

Backend tests cover:

- nested model defaults and SQL document persistence;
- disabled Agent prompt isolation;
- enabled Agent prompt and manifest injection;
- main-Agent and subagent consistency;
- absence of provider-owned code flags and environment injection;
- manifest endpoint 404, 409, 503, and success behavior;
- `AGENT_CODE_PATH` override guidance with `/opt/code` fallback.

Frontend tests cover:

- new Agents defaulting to code access off;
- the manual toggle persisting `code.enabled`;
- conditional manifest editor visibility;
- manifest generation persisting `code.manifest`;
- `/opt/code` as the displayed default path.

Full backend tests, frontend tests, lint, production build, and whitespace checks
must pass before delivery.
