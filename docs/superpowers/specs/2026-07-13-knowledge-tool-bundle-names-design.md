# Knowledge Tool Bundle and Naming Design

## Goal

Make the Agent knowledge capability an indivisible, clearly namespaced tool bundle. When knowledge is enabled for an Agent, all knowledge retrieval and inspection tools are callable. Knowledge tools must not be confused with shell or ordinary filesystem operations.

## Canonical Tool Names

The only supported names after this change are:

- `knowledge_search`: semantic, keyword, or hybrid retrieval across accessible knowledge bases.
- `knowledge_read`: read a complete knowledge document or locate a retrieved chunk in its surrounding document.
- `knowledge_find`: find an exact literal string within accessible knowledge-base content.
- `knowledge_list`: list accessible knowledge bases when the Agent needs to inspect or narrow scope.

The legacy names `view_file`, `grep_file`, and `list_knowledge_bases` are removed directly. They are not registered as runtime aliases and are never advertised to the model.

## Capability Semantics

Knowledge is one Agent-facing capability, not four independently selectable tools. The canonical knowledge bundle contains all four names.

The backend is authoritative:

- When an Agent enables knowledge, tool selection expands it to the complete canonical bundle.
- When knowledge is disabled or explicitly excluded, none of the bundle is exposed.
- A partial include containing any legacy or canonical knowledge tool is normalized to the complete bundle when configuration is loaded or saved.
- Runtime selection must not rely on the frontend having written the correct list.

The frontend knowledge toggle reads and writes the complete bundle. Default Agent configuration includes the complete bundle.

## Configuration Migration

Existing persisted Agent profiles are normalized without compatibility aliases at runtime:

- `view_file` becomes `knowledge_read`.
- `grep_file` becomes `knowledge_find`.
- `list_knowledge_bases` becomes `knowledge_list`.
- If any knowledge tool is enabled, all four canonical tools are added.
- All legacy names are removed from both include and exclude lists.
- If knowledge was explicitly disabled, normalization preserves the disabled state and does not re-enable the bundle.

Normalization is idempotent so repeated config loads and saves produce the same document.

## Prompt and Tool Contract

All system guidance, tool descriptions, tool-result instructions, subagent prompts, shell collision guards, and skill permission mappings use only canonical names.

The intended flow is:

1. `knowledge_search` retrieves globally ranked passages.
2. If evidence is incomplete, ambiguous, or lacks context, the Agent uses `knowledge_read` with the internal `document_id` or `chunk_id`.
3. If an exact identifier, error code, API name, or literal phrase must be located, the Agent uses `knowledge_find`.
4. `knowledge_list` is used only when discovery or explicit KB narrowing is useful; normal search still uses the Agent's configured KB scope or all accessible KBs.

Internal document and chunk IDs remain tool inputs and are never shown in the final answer.

## Runtime Observability

At run construction, debug-level diagnostics should make the effective state inspectable without logging document content:

- selected Agent ID;
- effective knowledge tool names;
- configured KB IDs;
- whether Agent-level rerank is enabled.

Existing `knowledge_search` invocation logs continue to include permission-resolved KB IDs.

## Error and Boundary Behavior

- Missing or unavailable `KnowledgeService`: no knowledge tools are registered; prompts must not advertise them.
- Knowledge capability disabled: the entire bundle is absent.
- Explicit Agent exclusion: the entire bundle is absent, even if an old config excludes only one legacy member.
- No configured KB IDs: search all permission-accessible KBs, preserving current behavior.
- No accessible KBs or no results: return the existing model-readable empty-result message without attempting `knowledge_read`.
- Invalid document/chunk access: `knowledge_read` returns a permission-safe error and never leaks existence across users.

## Testing

Backend tests cover:

- default Agent receives all four canonical tools;
- enabling knowledge expands to the complete bundle;
- disabling/excluding knowledge removes the complete bundle;
- old include/exclude configurations normalize correctly and idempotently;
- registry and stable prompt expose no legacy names;
- search output recommends `knowledge_read` and `knowledge_find` only;
- subagent knowledge profiles receive the complete bundle.

Frontend tests cover:

- the knowledge toggle writes all four canonical names;
- an Agent with any canonical bundle member is displayed as knowledge-enabled;
- disabling knowledge removes the complete bundle and legacy names.

A repository-wide stale-name scan must find no executable references to `view_file`, `grep_file`, or `list_knowledge_bases`; historical design documents may retain old names as historical context.

## Out of Scope

- Changing retrieval, reranking, permissions, or scoring behavior.
- Keeping runtime aliases for legacy tool names.
- Forcing a knowledge search on every conversational turn. Deterministic proactive-search policy can be designed separately from making the complete bundle available.
