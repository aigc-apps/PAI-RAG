# Knowledge Search Pipeline Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Agent rerank candidate limits effective, keep production UI on Elasticsearch hybrid retrieval, and expose privacy-safe spans for every important online knowledge-search stage.

**Architecture:** `KnowledgeService.search` remains the orchestration boundary: it retrieves per knowledge base, merges candidates, diversifies by document, truncates the global rerank pool, and reranks once. A new optional tracing adapter isolates OpenTelemetry imports from the knowledge service. Elasticsearch hybrid remains one request and is represented by one retrieval span with BM25 and kNN events.

**Tech Stack:** Python 3.11, FastAPI/SQLModel, Elasticsearch 8.x client, OpenTelemetry, pytest; React 18, TypeScript, Vitest/Testing Library; Markdown.

## Global Constraints

- Elasticsearch `hybrid` remains one `_search` request containing BM25 and kNN; do not add application-level fusion.
- Backend modes `hybrid`, `vector`, and `keyword` remain API-compatible; frontend requests and persisted defaults use `hybrid`.
- `top_k` defaults to 10 and is normalized to 1–50; each knowledge base retrieves `max(20, top_k + offset)` candidates.
- Agent `candidate_pool_size` defaults to 50, is configured in the 1–200 range, and its effective value is `max(candidate_pool_size, top_k + offset)`.
- With rerank enabled, diversify first using at most 3 chunks per `(kb_id, document_id)`, then truncate the global candidate pool, then rerank once.
- With rerank disabled, candidate diversification and `candidate_pool_size` do not affect retrieval ordering.
- Trace attributes must not contain raw query text, chunk text, titles, exception messages, or other knowledge content.
- Tracing remains optional: missing or disabled OpenTelemetry must never change search behavior.

---

## File Structure

- Create `backend/app/knowledge_tracing.py`: optional, privacy-safe span adapter used by the online search pipeline.
- Modify `backend/app/knowledge.py`: candidate-pool semantics and stage-level tracing orchestration.
- Modify `backend/tests/test_knowledge_retrieval_models.py`: candidate-pool, fallback, and span behavior tests.
- Modify `backend/tests/test_search_engine.py`: explicit Elasticsearch BM25/kNN request-window assertions.
- Modify `frontend/src/components/KnowledgeView.tsx`: remove mode state and controls; always submit `hybrid`.
- Modify `frontend/src/components/__tests__/KnowledgeView.test.tsx`: assert absent mode controls and hybrid requests.
- Modify `frontend/src/i18n/en.ts` and `frontend/src/i18n/zh.ts`: remove mode-dependent recall copy.
- Create `docs/agent/knowledge-search.md`: maintainer-facing search configuration, execution, fallback, and tracing guide.

### Task 1: Enforce the global rerank candidate pool

**Files:**
- Modify: `backend/tests/test_knowledge_retrieval_models.py`
- Modify: `backend/tests/test_search_engine.py`
- Modify: `backend/app/knowledge.py`

**Interfaces:**
- Consumes: `KnowledgeService.search(..., top_k: int, offset: int, rerank_config: Optional[dict]) -> tuple[list[SearchHit], int]`.
- Produces: rerank input ordered as `retrieve -> diversify(max_chunks=3) -> truncate(effective_pool)`; no public signature changes.

- [ ] **Step 1: Write failing candidate-pool tests**

Add focused tests using `RecordingSearchEngine` and `FakeReranker`:

```python
def test_rerank_diversifies_before_applying_candidate_pool():
    async def scenario():
        router = _chat_router()
        reranker = FakeReranker()
        router.register_llm("dashscope/rr", reranker)
        engine = RecordingSearchEngine({})
        svc = await _svc(router)
        svc._search = engine
        svc._fallback_to_local = False
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        engine.hits_by_kb[kb.id] = [
            _hit(
                kb_id=kb.id,
                document_id="long",
                chunk_id=f"long-{i}",
                title="Long",
                text=f"long {i}",
                score=1 - i / 100,
            )
            for i in range(6)
        ] + [
            _hit(
                kb_id=kb.id,
                document_id=f"other-{i}",
                chunk_id=f"other-{i}",
                title=f"Other {i}",
                text=f"other {i}",
                score=0.8 - i / 100,
            )
            for i in range(4)
        ]
        await svc.search(
            user=ADMIN,
            kb_ids=[kb.id],
            query="q",
            top_k=2,
            rerank_config={
                "enabled": True,
                "model": "dashscope/rr",
                "candidate_pool_size": 5,
            },
        )
        return reranker.calls[-1][1]

    documents = asyncio.run(scenario())
    assert len(documents) == 5
    assert sum("Document: Long" in document for document in documents) == 3
    assert any("Document: Other" in document for document in documents)


def test_candidate_pool_is_at_least_page_window():
    async def scenario():
        router = _chat_router()
        reranker = FakeReranker()
        router.register_llm("dashscope/rr", reranker)
        engine = RecordingSearchEngine({})
        svc = await _svc(router)
        svc._search = engine
        svc._fallback_to_local = False
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        engine.hits_by_kb[kb.id] = [
            _hit(
                kb_id=kb.id,
                document_id=f"doc-{i}",
                chunk_id=f"chunk-{i}",
                title=f"Doc {i}",
                text=f"body {i}",
                score=1 - i / 100,
            )
            for i in range(10)
        ]
        await svc.search(
            user=ADMIN,
            kb_ids=[kb.id],
            query="q",
            top_k=3,
            offset=4,
            rerank_config={
                "enabled": True,
                "model": "dashscope/rr",
                "candidate_pool_size": 2,
            },
        )
        return reranker.calls[-1]

    _query, documents, top_n = asyncio.run(scenario())
    assert len(documents) == 7
    assert top_n == 7


def test_rerank_disabled_does_not_apply_candidate_pool_or_document_cap():
    async def scenario():
        engine = RecordingSearchEngine({})
        svc = await _svc()
        svc._search = engine
        svc._fallback_to_local = False
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        engine.hits_by_kb[kb.id] = [
            _hit(
                kb_id=kb.id,
                document_id="same",
                chunk_id=f"chunk-{i}",
                title="Same",
                text=f"body {i}",
                score=1 - i / 100,
            )
            for i in range(5)
        ]
        return await svc.search(
            user=ADMIN,
            kb_ids=[kb.id],
            query="q",
            top_k=5,
            rerank_config={"enabled": False, "candidate_pool_size": 1},
        )

    hits, _total = asyncio.run(scenario())
    assert [hit.chunk_id for hit in hits] == [f"chunk-{i}" for i in range(5)]
```

Extend the existing Elasticsearch hybrid DSL test:

```python
assert body["size"] == 5
assert body["knn"]["k"] == 5
assert body["knn"]["num_candidates"] == 50
```

Add a second call with `limit=20` and assert `k == 20` and `num_candidates == 80`.

- [ ] **Step 2: Run the focused tests and verify the new pool assertions fail**

Run:

```bash
cd backend
pytest -q tests/test_knowledge_retrieval_models.py -k "candidate_pool or diversifies_before" \
  tests/test_search_engine.py::test_es_hybrid_search_dsl_has_knn_and_bm25
```

Expected: the candidate-pool tests fail because `_rerank_hits` currently receives every diversified candidate; existing Elasticsearch assertions pass.

- [ ] **Step 3: Implement candidate-pool normalization and truncation**

In `KnowledgeService.search`, compute these values after `rerank_on`:

```python
configured_candidate_pool = max(
    1, min(int(rerank_cfg.get("candidate_pool_size") or 50), 200)
)
effective_candidate_pool = max(configured_candidate_pool, limit + offset)
```

Change the rerank branch to preserve the required ordering:

```python
if rerank_on and candidates:
    candidates = self._cap_chunks_per_document(candidates, max_chunks=3)
    candidates = candidates[:effective_candidate_pool]
    candidates = await self._rerank_hits(
        query,
        candidates,
        limit + offset,
        rerank_cfg,
        kb_names={kb.id: kb.name for kb in allowed},
    )
```

Do not apply either operation outside the rerank branch.

- [ ] **Step 4: Run backend retrieval tests**

Run:

```bash
cd backend
pytest -q tests/test_knowledge_retrieval_models.py tests/test_search_engine.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the candidate-pool change**

```bash
git add backend/app/knowledge.py backend/tests/test_knowledge_retrieval_models.py backend/tests/test_search_engine.py
git commit -m "fix(knowledge): enforce rerank candidate pool"
```

### Task 2: Add optional stage-level knowledge search tracing

**Files:**
- Create: `backend/app/knowledge_tracing.py`
- Modify: `backend/tests/test_knowledge_retrieval_models.py`
- Modify: `backend/app/knowledge.py`

**Interfaces:**
- Produces: `knowledge_span(name: str, attributes: Optional[dict[str, AttributeValue]] = None) -> ContextManager[Optional[Span]]`.
- Produces: `set_span_attributes(span, attributes)`, `add_span_event(span, name, attributes=None)`, and `mark_span_error(span, error)`; every helper is no-op safe.
- Consumes: optional `extensions.trace.tracer.get_tracer`; callers never import OpenTelemetry directly.

- [ ] **Step 1: Write failing tracing-adapter and pipeline tests**

Create recording test doubles in `test_knowledge_retrieval_models.py`:

```python
class RecordingSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, dict(attributes or {})))


class RecordingSpanContext:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, *_args):
        return False
```

Monkeypatch `knowledge_tracing._get_tracer` to return a recording tracer and assert a hybrid, reranked multi-KB search produces:

```python
assert names.count("knowledge.search") == 1
assert names.count("knowledge.query_embedding") == 1
assert names.count("knowledge.retrieve.hybrid") == 2
assert names.count("knowledge.candidate_diversify") == 1
assert names.count("knowledge.rerank") == 1
assert {name for name, _ in retrieve_spans[0].events} == {"bm25", "vector_knn"}
assert root.attributes["knowledge.candidate_pool.configured"] == 5
assert root.attributes["knowledge.candidate_pool.effective"] == 5
assert "query" not in " ".join(root.attributes)
```

Add these explicit assertions to focused tests:

```python
def test_keyword_search_uses_bm25_span(recording_tracer):
    hits = run_recorded_search(mode="keyword", tracer=recording_tracer)
    span = next(
        span
        for span in recording_tracer.spans
        if span.name == "knowledge.retrieve.bm25"
    )
    assert hits
    assert span.events == []


def test_primary_fallback_updates_retrieval_span(recording_tracer):
    hits = run_failing_primary_search(tracer=recording_tracer)
    span = next(
        span
        for span in recording_tracer.spans
        if span.name == "knowledge.retrieve.bm25"
    )
    assert hits
    assert span.attributes["knowledge.fallback"] is True
    assert span.attributes["knowledge.engine"] == "local"
    assert span.attributes["error.type"] == "RuntimeError"


def test_trace_attributes_redact_search_content(recording_tracer):
    sentinel = "SENSITIVE_TRACE_VALUE_42"
    run_failing_embedding_and_rerank(
        sentinel=sentinel, tracer=recording_tracer
    )
    serialized = repr(
        [(span.attributes, span.events) for span in recording_tracer.spans]
    )
    assert sentinel not in serialized
    assert "RuntimeError" in serialized


def test_missing_trace_extension_does_not_change_results(monkeypatch):
    monkeypatch.setattr(
        knowledge_tracing,
        "_get_tracer",
        lambda: (_ for _ in ()).throw(ImportError("not installed")),
    )
    hits = run_recorded_search(mode="keyword")
    assert hits
```

Implement `recording_tracer`, `run_recorded_search`, `run_failing_primary_search`, and
`run_failing_embedding_and_rerank` as local test helpers using the existing `_svc`,
`RecordingSearchEngine`, `FailingEmbedder`, and `FailingReranker` seams; each helper
returns only hits or spans and never stores the raw sentinel in trace attributes.

- [ ] **Step 2: Run tracing tests and verify import/helper failures**

Run:

```bash
cd backend
pytest -q tests/test_knowledge_retrieval_models.py -k "trace or span or candidate_pool"
```

Expected: FAIL because `app.knowledge_tracing` and the stage spans do not exist.

- [ ] **Step 3: Implement the optional tracing adapter**

Create `backend/app/knowledge_tracing.py` with a guarded tracer lookup and null context:

```python
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _get_tracer():
    from extensions.trace.tracer import get_tracer

    return get_tracer()


@contextmanager
def knowledge_span(
    name: str, attributes: Optional[dict[str, Any]] = None
) -> Iterator[Any | None]:
    try:
        span_cm = _get_tracer().start_as_current_span(name)
    except Exception:
        yield None
        return
    with span_cm as span:
        set_span_attributes(span, attributes or {})
        yield span


def set_span_attributes(span, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            try:
                span.set_attribute(key, value)
            except Exception:
                pass


def add_span_event(
    span, name: str, attributes: Optional[dict[str, Any]] = None
) -> None:
    if span is not None:
        try:
            span.add_event(name, attributes=attributes or {})
        except Exception:
            pass


def mark_span_error(span, error: Exception) -> None:
    set_span_attributes(
        span, {"error.type": type(error).__name__, "knowledge.status": "error"}
    )
    try:
        from opentelemetry.trace import Status, StatusCode

        if span is not None:
            span.set_status(Status(StatusCode.ERROR))
    except Exception:
        pass
```

Never record `str(error)`.

- [ ] **Step 4: Instrument `KnowledgeService.search` and `_rerank_hits`**

Import the adapter functions into `knowledge.py`. Wrap the normalized search body in `knowledge.search`; update its aggregate attributes before returning. For each embedding group, create `knowledge.query_embedding` with provider/model/dimension/KB count and mark failures with `mark_span_error` while retaining current fallback.

For each KB, select the retrieval name exactly:

```python
retrieval_span_name = {
    "keyword": "knowledge.retrieve.bm25",
    "vector": "knowledge.retrieve.vector",
}.get(mode, "knowledge.retrieve.hybrid")
```

Set KB id, primary engine, requested limit, fallback flag, result count, and total. For hybrid add `bm25` and `vector_knn` events; set `knowledge.knn.k=per_kb_fetch_limit` and `knowledge.knn.num_candidates=max(50, per_kb_fetch_limit * 4)`. If local fallback runs, update engine to `local`, set fallback true and record only the primary exception type.

Wrap `_cap_chunks_per_document` with `knowledge.candidate_diversify`; set input/output/dropped counts and max chunks 3. Wrap reranker resolution and invocation with `knowledge.rerank`; set model, input count, requested top_n, output count, and `knowledge.status` (`ok`, `skipped`, or `fallback`). Mark provider exceptions without changing `_rerank_hits` result semantics.

- [ ] **Step 5: Run tracing and regression tests**

Run:

```bash
cd backend
pytest -q tests/test_knowledge_retrieval_models.py tests/test_knowledge_tool.py tests/test_routes_knowledge.py
```

Expected: all tests pass and sentinel privacy assertions remain green.

- [ ] **Step 6: Commit tracing**

```bash
git add backend/app/knowledge_tracing.py backend/app/knowledge.py backend/tests/test_knowledge_retrieval_models.py
git commit -m "feat(knowledge): trace online search stages"
```

### Task 3: Remove frontend retrieval-mode controls

**Files:**
- Modify: `frontend/src/components/__tests__/KnowledgeView.test.tsx`
- Modify: `frontend/src/components/KnowledgeView.tsx`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`

**Interfaces:**
- Consumes: `searchKnowledge(payload)` where backend mode remains a union.
- Produces: every recall-test request includes `mode: "hybrid"`; every saved `default_retrieval_config` includes `mode: "hybrid"`.

- [ ] **Step 1: Write failing UI tests**

Extend the recall test:

```typescript
expect(screen.queryByText(/^mode$/i)).not.toBeInTheDocument();
expect(screen.queryByRole("button", { name: "vector" })).not.toBeInTheDocument();
expect(searchKnowledge).toHaveBeenCalledWith(expect.objectContaining({ mode: "hybrid" }));
expect(screen.getByText(/showing first 1/i)).toBeInTheDocument();
expect(screen.queryByText(/mode hybrid/i)).not.toBeInTheDocument();
```

Add a config-tab test with a KB whose stored mode is `vector`; change `top_k`, save, and assert:

```typescript
expect(updateKnowledgeBase).toHaveBeenCalledWith(
  "kb_1",
  expect.objectContaining({
    default_retrieval_config: expect.objectContaining({ mode: "hybrid" }),
  }),
);
```

Mock `updateKnowledgeBase` in the existing API mock.

- [ ] **Step 2: Run the UI test and verify mode assertions fail**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx
```

Expected: FAIL because both mode selectors render and the recall request follows mode state.

- [ ] **Step 3: Remove mode state and fixed-mode copy**

In `retrievalOf`, remove the returned `mode`. In config state, dirty/reset logic, and JSX, remove `mode`/`setMode` and the mode `Seg`. Save with:

```typescript
default_retrieval_config: {
  mode: "hybrid",
  top_k: topK,
  score_threshold: threshold,
  force_citation: forceCite,
},
```

In `RecallPanel`, remove mode state and selector; call:

```typescript
return searchKnowledge({
  kb_ids: [kb.id], query, mode: "hybrid", top_k: topK,
  offset, score_threshold: threshold, filters,
});
```

Change `kbview.hitCountB` to omit `{mode}` and change `kbview.noHitsHint` to suggest relaxing the threshold or checking indexing only. Remove `kbview.mode` if no remaining component uses it.

- [ ] **Step 4: Run frontend tests and type/build checks**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx
npm test -- --run
npm run build
```

Expected: all Vitest tests pass and Vite production build succeeds.

- [ ] **Step 5: Commit the frontend behavior**

```bash
git add frontend/src/components/KnowledgeView.tsx \
  frontend/src/components/__tests__/KnowledgeView.test.tsx \
  frontend/src/i18n/en.ts frontend/src/i18n/zh.ts
git commit -m "fix(ui): standardize knowledge recall on hybrid search"
```

### Task 4: Document and verify the production search pipeline

**Files:**
- Create: `docs/agent/knowledge-search.md`
- Modify only if an existing suitable index link exists: `README.md`

**Interfaces:**
- Documents the exact runtime behavior implemented by Tasks 1–3; no code interface changes.

- [ ] **Step 1: Write the maintainer guide**

Create `docs/agent/knowledge-search.md` with these exact sections:

```markdown
# 知识库在线搜索

## 默认行为
生产 UI 固定使用 hybrid；后端仍保留 hybrid/vector/keyword。

## 参数关系
top_k=10；per_kb_fetch_limit=max(20, top_k+offset)；
effective_candidate_pool_size=max(candidate_pool_size, top_k+offset)。

## 执行顺序
权限过滤 → embedding 分组 → 每 KB 独立召回 → 全局排序 →
每文档最多 3 切片 → candidate pool → 统一 rerank → 全局分页。

## Elasticsearch hybrid
一个请求同时包含 multi_match BM25 和 dense_vector kNN；
k=offset+limit，num_candidates=max(50, (offset+limit)*4)。

## Rerank 与候选多样化
解释两者不可互相替代，以及 rerank 关闭时不应用候选池。

## 降级与恢复
解释 embedding、Elasticsearch、rerank 失败行为和本地搜索降级。

## Trace
列出 knowledge.search、query_embedding、retrieve.*、
candidate_diversify、rerank，以及 hybrid 的 bm25/vector_knn 事件；
说明属性不包含 query 或正文。

## 调优示例
给出 top_k=10、两个 KB、candidate_pool_size=50 时的数字化流程。
```

If `README.md` has a documentation list, add a single link to this guide; otherwise do not restructure the README.

- [ ] **Step 2: Validate documentation and changed-code formatting**

Run:

```bash
rg -n "top_k|candidate_pool_size|per_kb_fetch_limit|knowledge\.retrieve" docs/agent/knowledge-search.md
git diff --check
cd backend && ruff check app/knowledge.py app/knowledge_tracing.py \
  tests/test_knowledge_retrieval_models.py tests/test_search_engine.py
```

Expected: each runtime concept is present, `git diff --check` has no output, and Ruff passes.

- [ ] **Step 3: Run final backend and frontend verification**

Run:

```bash
cd backend
pytest -q
cd ../frontend
npm test -- --run
npm run build
```

Expected: backend suite passes (apart from explicitly pre-existing skips/warnings), frontend suite passes, and production build succeeds.

- [ ] **Step 4: Review scope and working tree**

Run:

```bash
git status --short
git diff --stat
git diff -- backend/app/knowledge.py backend/app/knowledge_tracing.py \
  frontend/src/components/KnowledgeView.tsx docs/agent/knowledge-search.md
```

Expected: only the planned knowledge-search files plus pre-existing Agent code configuration work and user-owned `frontend/next-env.d.ts` are present; no unrelated files are staged.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/agent/knowledge-search.md README.md
git commit -m "docs: explain knowledge search pipeline"
```

Omit `README.md` from `git add` when no index link was needed.
