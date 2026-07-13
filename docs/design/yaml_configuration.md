# PAI-Loop YAML Configuration

This document describes the admin YAML edited from Settings -> YAML. In
service deployments, SQL is the runtime source of truth. YAML is the import/export
format and bootstrap seed for first startup. Runtime health is computed by the
backend and should not be written by hand.

## Storage Model

PAI-Loop stores authored product configuration in SQL:

- `app_config_documents`: singleton current document (`id: default`)
- `app_config_revisions`: append-only revision history

On first startup, if SQL has no config document, the backend imports the YAML
seed at `CONFIG_PATH` (default `./data/config.yaml`) once. After that, Settings
reads and writes SQL. Editing the seed file does not change a running deployment.

The YAML Settings page remains useful for advanced editing, backup, review, and
migration:

- `GET /v1/config.yaml` exports the current SQL config as YAML.
- `PUT /v1/config.yaml` validates YAML and saves it back to SQL.
- Local YAML files are no longer the multi-instance source of truth.

## Ownership Model

- `models` configures model providers, registered models, and defaults.
- `knowledgebase.vectordb` configures the global vector database used by the
  Knowledge capability.
- `skills` configures the skill library, installation policy, dependency policy,
  and installed skill records.
- `default_instructions` is the template copied into newly created agents.
- `agents` defines agent profiles and which tools or skills each agent may use.
- `providers` configures external service backends for core capabilities such as
  search, sandbox, and cloud authorization.
- `capabilities` configures system-level capabilities and installed skills.

The Knowledge tab is for knowledge-base data management. Vector DB connection
settings belong under the Knowledge capability, backed by
`knowledgebase.vectordb`.

## Do Not Author Runtime Fields

These fields are derived at runtime and are removed from persisted YAML:

- `providers[].status`
- `providers[].secret_configured`
- `providers[].error`
- `capabilities[].status`
- `capabilities[].error`
- `knowledgebase.vectordb.status`
- `knowledgebase.vectordb.secret_configured`
- `knowledgebase.vectordb.error`

The API still returns runtime status in structured config responses where the UI
needs it, but the raw YAML editor shows the authored configuration shape.

Exception: `skills.installed[].status` and `skills.installed[].dependency_status`
are installation records for uploaded skill packages. They are persisted metadata,
not capability health fields.

## Legacy Fields

These provider slots are obsolete and should not appear in YAML:

- `embedding.default`
- `rerank.default`
- `vectordb.default`

Embedding and rerank defaults now live in `models.default_embedding_model` and
`models.default_rerank_model`. The vector database now lives in
`knowledgebase.vectordb`. The Knowledge capability should not reference those
legacy provider ids.

## Models

Example:

```yaml
models:
  default_model: dashscope/qwen3.7-max
  default_embedding_model: dashscope/text-embedding-v4
  default_rerank_model: dashscope/qwen3-rerank
  providers:
    - name: dashscope
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key_env: DASHSCOPE_API_KEY
      models:
        - id: qwen3.7-max
          context_window: 988000
          max_output_tokens: 16384
          supports_tools: true
        - id: text-embedding-v4
          type: embedding
          protocol: dashscope
          dimension: 1024
        - id: qwen3-rerank
          type: rerank
          protocol: dashscope
```

Model provider API keys should be referenced by environment variable name through
`api_key_env`; do not put model API keys inline in YAML.

## Knowledge Vector Database

Example:

```yaml
knowledgebase:
  vectordb:
    engine: elasticsearch
    url: https://example.elasticsearch.aliyuncs.com:9200
    index_prefix: kb
    api_key: ""
    api_key_env: ""
    username: elastic
    password: ""
    password_env: ELASTICSEARCH_PASSWORD
    verify_certs: true
    timeout: 30
```

Use either API-key auth or username/password auth. Prefer `api_key_env` or
`password_env` over inline secrets. Existing knowledge bases may need reindexing
after changing the vector database.

## Providers

`providers` describes backend services used by core capabilities. Keep only
authored connection/settings fields here.

Typical provider ids:

- `llm.default`: runtime view of the selected default model.
- `search.default`: web search provider settings.
- `sandbox.default`: remote Agent Loop sandbox settings.
- `aliyun_pai.default`: base credential env names for Aliyun PAI authorization.

Provider health is computed by the backend and should not be authored in YAML.

## Capabilities

Example:

```yaml
capabilities:
  - id: knowledge
    kind: core_tool
    name: Knowledge Base
    description: Retrieve from local documents, with optional cloud-enhanced retrieval.
    enabled: true
    permission: auto
    dependencies: []
    provider_refs: []
    settings:
      mode: local
```

`install_skill` and `enable_skill_for_agent` are control-plane tools. They remain
hidden from ordinary capability configuration and are not user-facing agent
skills.

## Agents

Agent profiles define the visible agent name, optional model override, persona,
knowledge scope, and enabled tools/skills.

For PAI-Loop, sandbox-backed file and command tools run remotely. Agents should
not assume local host command execution.

```yaml
agents:
  - id: main
    name: MiniAgent
    model: ""
    instructions: ""
    knowledge:
      kb_ids: []
    tools:
      include:
        - current_datetime
        - web_fetch
        - web_search
        - knowledge_search
        - code_interpreter
        - shell
      exclude: []
    skills:
      enabled: []
    settings:
      max_steps: 20
```

An empty `model` means the agent inherits `models.default_model`. An empty
`instructions` value means the built-in persona is used.
