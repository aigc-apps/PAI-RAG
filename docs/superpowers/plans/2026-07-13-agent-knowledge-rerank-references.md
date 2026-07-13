# Agent Knowledge Rerank and References Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move rerank policy to each Agent, search selected knowledge bases as one candidate pool, simplify recall-test scoring, log resolved search scope, and expose only document titles plus accessible links in answers.

**Architecture:** `AgentKnowledgeConfig` owns the rerank model and candidate-pool size. The builder threads this policy through `ToolScope`; `knowledge_search` resolves KBs and passes the policy to `KnowledgeService.search`, which retrieves compatible embedding groups, merges candidates, and optionally reranks the merged pool once. KB recall-test requests omit Agent rerank and display only the returned final score.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLModel, Loguru, pytest; React 19, TypeScript, Zustand, Vitest, Testing Library.

## Global Constraints

- User-visible references contain a document title and an HTTP/HTTPS link when available; without an accessible URL they contain only the title.
- Never expose `[n]`, `document_id`, `chunk_id`, internal paths, or non-web URIs in the final answer.
- When a chunk is incomplete, ambiguous, or insufficient, instruct the Agent to use `view_file` and, when useful, `grep_file` before answering.
- Multiple KBs form one candidate pool and receive at most one Agent-configured rerank call.
- Without an Agent reranker, or when it fails, order by the retrieval engine's final similarity score.
- KB recall tests never apply Agent rerank and show only one final score per result.
- Preserve current soft-default `kb_ids` semantics, explicit-argument precedence, and per-KB permission checks.
- Preserve persisted KB `rerank_config` only for backward data compatibility; stop reading or editing it in active search and KB UI.
- The worktree contains unrelated offline-pipeline changes. Stage only task-specific hunks and never reset unrelated modifications.

---

### Task 1: Add Agent-Level Rerank Configuration

**Files:**
- Modify: `backend/app/agent_config.py`
- Modify: `frontend/src/api/agentConfig.ts`
- Modify: `frontend/src/components/SettingsView.tsx`
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Test: `backend/tests/test_routes_config.py`

**Interfaces:**
- Produces `AgentKnowledgeRerankConfig(enabled: bool, model: str, candidate_pool_size: int)`.
- Produces matching TypeScript `AgentKnowledgeRerankConfig`.
- Later tasks consume `agent.knowledge.rerank`; no database migration is needed because this is configuration-document data.

- [ ] **Step 1: Write failing backend configuration tests**

```python
def test_agent_knowledge_rerank_defaults_and_round_trips(tmp_path):
    path = tmp_path / "config.yaml"
    doc = load_agent_config(path)
    assert doc.agents[0].knowledge.rerank.enabled is False
    assert doc.agents[0].knowledge.rerank.model == ""
    assert doc.agents[0].knowledge.rerank.candidate_pool_size == 50

    doc.agents[0].knowledge.rerank = AgentKnowledgeRerankConfig(
        enabled=True, model="dashscope/qwen3-rerank", candidate_pool_size=80
    )
    save_agent_config(path, doc)
    loaded = load_agent_config(path)
    assert loaded.agents[0].knowledge.rerank.model == "dashscope/qwen3-rerank"
    assert loaded.agents[0].knowledge.rerank.candidate_pool_size == 80
```

- [ ] **Step 2: Run the backend test and verify RED**

Run: `cd backend && uv run pytest tests/test_routes_config.py -q`

Expected: FAIL because `AgentKnowledgeRerankConfig` and `knowledge.rerank` do not exist.

- [ ] **Step 3: Implement backend configuration models**

```python
class AgentKnowledgeRerankConfig(BaseModel):
    enabled: bool = False
    model: str = ""
    candidate_pool_size: int = Field(default=50, ge=1, le=200)


class AgentKnowledgeConfig(BaseModel):
    kb_ids: List[str] = Field(default_factory=list)
    rerank: AgentKnowledgeRerankConfig = Field(
        default_factory=AgentKnowledgeRerankConfig
    )
```

Existing documents without `rerank` receive defaults through Pydantic.

- [ ] **Step 4: Run backend tests and verify GREEN**

Run: `cd backend && uv run pytest tests/test_routes_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Write the failing Agent settings test**

```tsx
it("stores rerank policy on the selected agent", async () => {
  const user = userEvent.setup();
  const save = vi.fn(async (doc: AgentConfigDocument) => doc);
  useAgentConfigStore.setState({ save });
  const doc = {
    ...baseDoc,
    models: {
      ...baseDoc.models,
      providers: [{ name: "dashscope", models: [{ id: "rr", type: "rerank" }] }],
    },
  };
  render(<SettingsView doc={doc} onBack={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "编辑知识库" }));
  await user.selectOptions(screen.getByLabelText("Rerank model"), "dashscope/rr");
  await user.clear(screen.getByLabelText("Candidate pool size"));
  await user.type(screen.getByLabelText("Candidate pool size"), "80");
  await user.tab();
  const saved = save.mock.calls.at(-1)?.[0] as AgentConfigDocument;
  expect(saved.agents[0].knowledge.rerank).toEqual({
    enabled: true, model: "dashscope/rr", candidate_pool_size: 80,
  });
});
```

- [ ] **Step 6: Run the UI test and verify RED**

Run: `cd frontend && npm test -- --run src/components/__tests__/SettingsView.test.tsx`

Expected: FAIL because the Agent knowledge dialog has no rerank controls.

- [ ] **Step 7: Implement frontend types and Agent controls**

```ts
export interface AgentKnowledgeRerankConfig {
  enabled: boolean;
  model: string;
  candidate_pool_size: number;
}
export interface AgentKnowledgeConfig {
  kb_ids: string[];
  rerank: AgentKnowledgeRerankConfig;
}
```

Seed new Agents with disabled rerank. In `KnowledgeSection`, list catalog models with `type === "rerank"`, render a disabled option plus candidate-pool input, and preserve `kb_ids` on every save:

```tsx
const rerank = agent.knowledge?.rerank ?? {
  enabled: false, model: "", candidate_pool_size: 50,
};
const nextKnowledge = {
  kb_ids: [...selected],
  rerank: {
    enabled: Boolean(model),
    model,
    candidate_pool_size: Math.max(1, Math.min(200, candidatePoolSize)),
  },
};
void onSave(applyAgentPatch(doc, agent.id, { knowledge: nextKnowledge }));
```

- [ ] **Step 8: Run Settings tests and verify GREEN**

Run: `cd frontend && npm test -- --run src/components/__tests__/SettingsView.test.tsx`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add backend/app/agent_config.py backend/tests/test_routes_config.py frontend/src/api/agentConfig.ts frontend/src/components/SettingsView.tsx frontend/src/components/__tests__/SettingsView.test.tsx
git commit -m "feat(knowledge): configure reranking per agent"
```

---

### Task 2: Search Compatible KB Groups and Rerank One Unified Pool

**Files:**
- Modify: `backend/app/knowledge.py`
- Test: `backend/tests/test_knowledge_retrieval_models.py`
- Test: `backend/tests/test_search_engine.py`

**Interfaces:**
- Adds an optional `rerank_config: Optional[dict] = None` keyword argument to the existing `KnowledgeService.search` signature.
- Produces `KnowledgeService.resolve_search_kbs(*, user: User, kb_ids: list[str]) -> list[KnowledgeBaseRow]` so callers can log the permission-filtered scope.
- `rerank_config=None` means retrieval-score ordering.
- Configured policy uses `model` and `candidate_pool_size`, calls one reranker for merged candidates, and returns relevance in `SearchHit.score`.
- REST callers remain compatible because the argument is optional.

- [ ] **Step 1: Write failing multi-KB tests**

Add tests proving:

```python
hits, _ = await svc.search(
    user=ADMIN,
    kb_ids=[kb_a.id, kb_b.id],
    query="turbox",
    top_k=2,
    rerank_config={
        "enabled": True,
        "model": "dashscope/rr",
        "candidate_pool_size": 20,
    },
)
assert len(reranker.calls) == 1
assert {hit.kb_id for hit in hits} == {kb_a.id, kb_b.id}
assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
```

Use KBs with different embedding model/dimension configurations and recording embedders to prove each compatible group embeds the query with its own configuration. Add tests proving KB-persisted `rerank_config` is ignored when the argument is absent, and reranker failure preserves retrieval-score ordering.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd backend && uv run pytest tests/test_knowledge_retrieval_models.py tests/test_search_engine.py -q`

Expected: FAIL because `search` does not accept Agent policy and still uses the first KB's embedder/rerank settings.

- [ ] **Step 3: Implement grouped retrieval and final ranking**

Add focused helpers:

```python
async def resolve_search_kbs(
    self, *, user: User, kb_ids: list[str]
) -> list[KnowledgeBaseRow]:
    rows: list[KnowledgeBaseRow] = []
    for kb_id in kb_ids:
        try:
            rows.append(await self.get_kb(kb_id, user=user))
        except PermissionError:
            continue
    return rows


def _embedding_group_key(self, kb: KnowledgeBaseRow) -> tuple[str, str, int]:
    cfg = kb.embedding_config or {}
    return (
        str(cfg.get("provider_id") or "local_hash"),
        str(cfg.get("model") or "local-hash-v1"),
        int(cfg.get("dimension") or 64),
    )
```

Change `search` to resolve allowed rows through `resolve_search_kbs`, group by embedding configuration, embed once per group, fetch candidates from each group, merge by retrieval `score`, optionally truncate to `candidate_pool_size` and call `_rerank_hits` once, then apply global offset/limit. Log candidate count, rerank model, final count, and fallback reason inside this layer, where those values are known. Do not read `KnowledgeBaseRow.rerank_config`. Preserve engine and reranker fallback behavior.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cd backend && uv run pytest tests/test_knowledge_retrieval_models.py tests/test_search_engine.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/knowledge.py backend/tests/test_knowledge_retrieval_models.py backend/tests/test_search_engine.py
git commit -m "feat(knowledge): rerank unified agent search results"
```

---

### Task 3: Thread Agent Policy into the Tool and Log Resolved KBs

**Files:**
- Modify: `backend/app/builder.py`
- Modify: `backend/agent/tools/scope.py`
- Modify: `backend/agent/tools/builtin/knowledge.py`
- Test: `backend/tests/test_builder.py`
- Test: `backend/tests/test_knowledge_tool.py`

**Interfaces:**
- Produces `scope_knowledge_rerank() -> Dict[str, Any]`.
- Builder writes `metadata["knowledge_rerank"]`.
- Tool passes the resolved policy through the `rerank_config` keyword argument of `KnowledgeService.search`.
- Resolved log uses prefix `Calling tool knowledge_search with args:` and contains `resolved_kb_ids`.

- [ ] **Step 1: Write failing scope, forwarding, and log tests**

```python
assert ctx.metadata["knowledge_rerank"] == {
    "enabled": True,
    "model": "dashscope/rr",
    "candidate_pool_size": 80,
}
```

Capture Loguru and assert:

```python
messages = []
sink = logger.add(messages.append, format="{message}")
try:
    await tool.fn(query="turbox", top_k=10)
finally:
    logger.remove(sink)
assert service.search_calls[-1]["kb_ids"] == [kb_a.id, kb_b.id]
assert service.search_calls[-1]["rerank_config"]["model"] == "dashscope/rr"
line = next(m for m in messages if "resolved_kb_ids" in m)
assert "Calling tool knowledge_search with args:" in line
assert kb_a.id in line and kb_b.id in line
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd backend && uv run pytest tests/test_builder.py tests/test_knowledge_tool.py -q`

Expected: FAIL because rerank metadata and resolved logging do not exist.

- [ ] **Step 3: Implement scope propagation and resolved logging**

```python
def scope_knowledge_rerank() -> Dict[str, Any]:
    raw = get_current_tool_scope().metadata.get("knowledge_rerank")
    return dict(raw) if isinstance(raw, dict) else {}
```

Serialize the selected Agent policy in both builder paths. In `knowledge_search`, resolve requested targets using existing precedence, filter them with `resolve_search_kbs`, then log and call search:

```python
rerank = scope_knowledge_rerank()
allowed = await knowledge_service.resolve_search_kbs(user=user, kb_ids=targets)
resolved_kb_ids = [kb.id for kb in allowed]
resolved_args = {
    "query": query.strip(),
    "top_k": top_k,
    "resolved_kb_ids": resolved_kb_ids,
    "rerank": rerank.get("model") if rerank.get("enabled") else None,
}
logger.info(f"Calling tool knowledge_search with args: {resolved_args}")
hits, _total = await knowledge_service.search(
    user=user,
    kb_ids=resolved_kb_ids,
    query=query.strip(),
    top_k=max(1, min(int(top_k or _DEFAULT_TOP_K), 20)),
    mode=mode if mode in {"hybrid", "vector", "keyword"} else "hybrid",
    rerank_config=rerank,
)
```

Log the final result count after search without logging document content; candidate count and fallback reason come from `KnowledgeService.search`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cd backend && uv run pytest tests/test_builder.py tests/test_knowledge_tool.py -q`

Expected: all tests pass and logs include resolved KB IDs.

- [ ] **Step 5: Commit**

```bash
git add backend/app/builder.py backend/agent/tools/scope.py backend/agent/tools/builtin/knowledge.py backend/tests/test_builder.py backend/tests/test_knowledge_tool.py
git commit -m "feat(knowledge): log resolved agent search scope"
```

---

### Task 4: Replace Chunk Citations with Document References and Evidence Escalation

**Files:**
- Modify: `backend/agent/soul.py`
- Modify: `backend/agent/tools/builtin/knowledge.py`
- Test: `backend/tests/test_soul.py`
- Test: `backend/tests/test_knowledge_tool.py`

**Interfaces:**
- Tool output retains internal locate IDs for follow-up tools.
- Prompt requires deduplicated title plus HTTP/HTTPS link, or title only.
- Prompt forbids user-visible chunk indexes and internal identifiers.

- [ ] **Step 1: Write failing prompt and tool-format tests**

```python
prompt = render_stable_system_prompt(
    PERSONA, tool_names=["knowledge_search", "view_file", "grep_file"]
)
assert (
    "incomplete" in prompt and "view_file" in prompt and "grep_file" in prompt
)
assert "document_id" in prompt and "chunk_id" in prompt
assert "[n]" not in prompt
assert "Document: 安装指南" in out
assert "document_id:" in out and "chunk_id:" in out
assert "[1]" not in out
assert "Cite sources by their [n]" not in out
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd backend && uv run pytest tests/test_soul.py tests/test_knowledge_tool.py -q`

Expected: FAIL because the tool emits `[1]` and asks for `[n]` citations.

- [ ] **Step 3: Implement document-oriented output and guidance**

```python
lines.append(f"Document: {title}")
if h.source_uri:
    lines.append(f"    source: {h.source_uri}")
lines.append(f"    document_id: {h.document_id}")
if h.chunk_id:
    lines.append(f"    chunk_id: {h.chunk_id}")
lines.append(f"    final_score: {h.score:.6f}")
lines.append(f"    passage: {text}")
```

Replace trailing guidance with: use IDs only for `view_file`/`grep_file`; never expose them; read surrounding content when evidence is incomplete; deduplicate references by document; show title plus HTTP/HTTPS URL, otherwise title only; do not use `[n]`. Put auxiliary-tool names in `_KNOWLEDGE_AUX_GUIDANCE` so unavailable tools are not mentioned.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cd backend && uv run pytest tests/test_soul.py tests/test_knowledge_tool.py -q`

Expected: all tests pass and output has no numbered citation labels.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/soul.py backend/agent/tools/builtin/knowledge.py backend/tests/test_soul.py backend/tests/test_knowledge_tool.py
git commit -m "fix(knowledge): cite documents instead of chunks"
```

---

### Task 5: Simplify KB Configuration and Recall Scores

**Files:**
- Modify: `frontend/src/components/KnowledgeView.tsx`
- Modify: `frontend/src/components/__tests__/KnowledgeView.test.tsx`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`

**Interfaces:**
- KB create/edit forms no longer read or submit rerank settings.
- Recall cards render only `KnowledgeHit.score` as localized final score.
- `searchKnowledge` remains unchanged and sends no Agent rerank policy.

- [ ] **Step 1: Write failing Knowledge UI tests**

Mock a recall result with `score=0.731`, `vector_score=0.912`, and `keyword_score=0.456`. Render the recall tab, execute a search, then assert:

```tsx
expect(await screen.findByText("0.731")).toBeInTheDocument();
expect(screen.getByText(/final score/i)).toBeInTheDocument();
expect(screen.queryByText(/vector score/i)).not.toBeInTheDocument();
expect(screen.queryByText(/keyword score/i)).not.toBeInTheDocument();
```

Add a test that KB config and create forms do not render a rerank model selector.

- [ ] **Step 2: Run the focused UI test and verify RED**

Run: `cd frontend && npm test -- --run src/components/__tests__/KnowledgeView.test.tsx`

Expected: FAIL because score bars and KB rerank controls remain.

- [ ] **Step 3: Remove KB rerank controls and render one final score**

Remove KB rerank state, dirty checks, patches, selectors, chips, and create-form fields. Keep embedding configuration. Replace score bars with:

```tsx
<span className="ml-auto inline-flex items-center gap-2 text-xs">
  <span className="text-[var(--text-faint)]">{t("kbview.finalScore")}</span>
  <span className="font-mono font-semibold text-[var(--accent)]">
    {hit.score.toFixed(3)}
  </span>
</span>
```

Delete the unused `ScoreBar`. Add translations `"kbview.finalScore": "Final score"` and `"kbview.finalScore": "最终得分"`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cd frontend && npm test -- --run src/components/__tests__/KnowledgeView.test.tsx`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/KnowledgeView.tsx frontend/src/components/__tests__/KnowledgeView.test.tsx frontend/src/i18n/en.ts frontend/src/i18n/zh.ts
git commit -m "fix(knowledge): show only final recall scores"
```

---

### Task 6: Full Verification

**Files:**
- Modify only if stale behavior is documented: `README.md`
- Modify only if stale behavior is documented: `backend/README.md`

**Interfaces:** No new interface; verify all prior deliverables together.

- [ ] **Step 1: Search for stale active behavior**

Run:

```bash
rg -n "Cite sources by their \[n\]|rerank_model|rerank_top_n|chipRerank|ScoreBar" backend frontend/src --glob '!frontend/node_modules/**' --glob '!frontend/dist/**'
```

Expected: no active Agent prompt or KB UI references. Legacy schema/storage references may remain when clearly compatibility-only.

- [ ] **Step 2: Run complete backend tests**

Run: `cd backend && uv run pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run backend lint and typing hooks**

```bash
pre-commit run ruff --files backend/app/agent_config.py backend/app/builder.py backend/app/knowledge.py backend/agent/soul.py backend/agent/tools/scope.py backend/agent/tools/builtin/knowledge.py backend/tests/test_routes_config.py backend/tests/test_builder.py backend/tests/test_knowledge_retrieval_models.py backend/tests/test_search_engine.py backend/tests/test_soul.py backend/tests/test_knowledge_tool.py
pre-commit run mypy --files backend/app/agent_config.py backend/app/builder.py backend/app/knowledge.py backend/agent/soul.py backend/agent/tools/scope.py backend/agent/tools/builtin/knowledge.py
```

Expected: both hooks pass.

- [ ] **Step 4: Run complete frontend tests and build**

```bash
cd frontend && npm test -- --run
cd frontend && npm run build
```

Expected: all tests pass and Vite completes the production build; the existing chunk-size warning is non-blocking.

- [ ] **Step 5: Review final diff**

```bash
git diff --check
git status --short
git diff --stat
```

If README text describes KB-level reranking, update only that text and commit:

```bash
git add README.md backend/README.md
git commit -m "docs: document agent knowledge reranking"
```

If neither README requires change, do not create an empty commit.
