# Agent Capabilities and Skills Model

## Purpose

PAI-Loop is an Agent Loop service, not a local desktop agent. The agent cannot
inspect the user's local filesystem or run local commands directly. Its runtime
actions happen through configured service channels such as web search, knowledge
retrieval, and the remote sandbox.

This document defines the product language and runtime mapping for Agent
configuration so users can understand what they are enabling without seeing
low-level tool names.

## Product Vocabulary

### Connections

Connections are deployment-level service configuration: model providers, web
search providers, sandbox credentials, vector database settings, and cloud
authorization.

Users configure a connection once. Agents then opt into capabilities backed by
those connections.

Examples:

- Model provider endpoint and API key
- Tavily or Brave web search API key
- AgentRun sandbox template and credentials
- Elasticsearch vector database
- Alibaba Cloud authorization

### Capabilities

Capabilities are the channels an agent can use. They answer "what can this agent
access or do?"

Capabilities are user-facing bundles over one or more low-level tools. The main
UI should show capability names, not implementation tool names.

Recommended user-facing capabilities:

- Web Search
- Knowledge Base
- Code Sandbox
- File Output
- Cloud Access

Example mapping:

```text
Web Search
  -> web_search
  -> web_fetch

Knowledge Base
  -> knowledge_search
  -> optional advanced tools: knowledge_read, knowledge_grep, knowledge_list

Code Sandbox
  -> code_interpreter
  -> shell
  -> future structured tools: sandbox_read_file, sandbox_grep

File Output
  -> publish_artifact

Cloud Access
  -> sandbox CLI access plus injected temporary cloud credentials
```

### Skills

Skills are task playbooks. They answer "how should this agent perform a specific
workflow?"

A skill should not replace a capability. It should teach the agent how to use
available capabilities in a reliable sequence.

Examples:

- Knowledge QA: how to answer from retrieved documentation and cite sources
- PAI-Rec Diagnosis: how to gather cloud state, inspect logs, validate configs,
  and report findings
- Report Generation: how to structure and export a report

Skills may depend on capabilities. For example, Knowledge QA depends on
Knowledge Base. PAI-Rec Diagnosis depends on Code Sandbox and Cloud Access.

### Knowledge Scope

Knowledge Scope is per-agent scoping over knowledge bases. It answers "which
knowledge bases should this agent search by default?"

It is not a capability and not a skill. The Knowledge Base capability determines
whether the agent can retrieve from knowledge at all. Knowledge Scope narrows the
default set of bases for one agent.

## Runtime Assembly

Runtime assembly starts from the selected agent profile.

```text
Agent profile
  -> tools.include / tools.exclude
  -> selected low-level tool names
  -> ToolBox exposed to the model
  -> capability prompt fragments gated by actual tool names
  -> enabled skill catalog
  -> load_skill on demand for full skill instructions
```

The backend should only inject capability prompts for capabilities that are
actually available in the selected agent's toolbox.

Examples:

- If `knowledge_search` is selected, inject Knowledge Base guidance.
- If `shell` or `code_interpreter` is selected, inject Code Sandbox guidance.
- If `publish_artifact` is selected, inject File Output guidance.
- If `web_search` is selected, inject Web Search guidance.
- If cloud authorization is configured and `shell` is selected, inject Cloud
  Access guidance.

This prevents the model from being instructed to call unavailable tools.

## Prompt Layers

Prompt content should be layered so each concept has one home.

```text
Persona prompt
  Describes who the agent is, tone, and general behavior.

Capability prompts
  Describe safe and effective use of currently available channels.

Skill prompts
  Describe task-specific workflows and are loaded on demand.

Volatile context
  User memory, conversation summary, and per-request instructions.
```

Persona prompts should not contain low-level tool protocol. Capability prompts
should not contain long task workflows. Skill prompts should not expose new
actions unless the required capability is available.

## Knowledge Tools

Knowledge tools and sandbox tools must remain conceptually separate.

Knowledge tools operate on indexed knowledge-base documents with retrieval,
permissions, and source metadata.

Sandbox tools operate on files and commands inside the remote sandbox mounts.

Recommended names:

```text
knowledge_search
knowledge_read
knowledge_grep
knowledge_list

sandbox_read_file
sandbox_grep
shell
code_interpreter
publish_artifact
```

The current implementation still has legacy names such as `view_file`,
`grep_file`, and `list_knowledge_bases`. These should be treated as advanced
knowledge-base inspection tools, not generic local file tools.

## UI Guidance

The default Settings UI should use these labels:

- Connections: configure service endpoints, credentials, and providers.
- Capabilities: choose channels an agent may use.
- Skills: choose task playbooks an agent may follow.
- Knowledge Scope: choose which knowledge bases an agent searches by default.

Avoid showing low-level names such as `shell`, `code_interpreter`,
`publish_artifact`, `view_file`, or `grep_file` in the main Agent settings UI.
Expose them only in advanced views, diagnostics, or YAML.

## Default Agent Recommendation

For the default PAI-Loop agent:

- Enable Code Sandbox by default because remote execution is central to the
  product.
- Enable Knowledge Base by default when a knowledge service is available.
- Enable Web Search when a provider is configured.
- Enable File Output when artifact publishing is configured.
- Keep Skills minimal by default; `Knowledge QA` is a reasonable default because
  it teaches grounded answers without adding a new execution channel.
