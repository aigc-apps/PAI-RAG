# 分层上下文管理 —— offload-not-truncate + 大负载 DB 分表 + 相关性召回 · 实现方案

**Date:** 2026-07-12
**Status:** Draft for review
**Branch:** `personal/yfei/agent-core`
**Builds on:** `agent/budgeting.py`（`fit_to_budget`/`cap_tool_result`）、`app/builder.py`（`build_context`/`items_to_messages`）、`api/protocol/responses_serializer.py`（`on_tool_result`）、`app/store/*`（`ResponseStore`，全量 tool_result 已在此）、`app/knowledge.py`（ES 混合检索）。
**参考:** `docs/design/harness-gap-analysis.md` §4-② / §5-③（本文为该项落地设计）。

---

## 1. Problem

顶尖 harness 的最重工程投入在**上下文管理**（Manus："缓存命中率是生产 agent 最重要的单一指标"，10x 成本差）。PAI-Loop 现为**单层**：一条 rolling summary（`app/summarizer.py`）+ `budgeting.fit_to_budget` 的窗口内压缩。长任务会"丢中段"，且工具结果一旦被压缩就无法精确回取。

## 2. 现状核查（2026-07-12，已验证）

**结论：存储层已经是无损的，源真相天然分布式；缺的是投影智能 + 大负载分表 + 检索。**

| 维度 | 现状 | 判定 |
|---|---|---|
| **tool_result 是否全量落库** | `ToolResult.output = result.content`（全量）→ `on_tool_result` 原样写 `function_call_output` → `append_items` 落 DB。`cap_tool_result` 只改活跃 run 的内存 `messages`，不落库。 | ✅ **全量、无损** |
| **能否无状态重建** | `resolve_history` → `items_to_messages`（`builder.py:68-74`）读 `output` 全量回填 `ctx.history`；worker 只需 `conversation_id`。 | ✅ **已分布式** |
| **窗口投影** | `fit_to_budget`：L1 截断 tool_round → L2 摘要 → L3 丢整组 → 截断 protected history；**全在窗口内、纯位置式、不写回**。 | 🟡 **有损且不可回取** |
| **新 vs 历史 tool_result 不对称** | 新产生的走 `cap_tool_result`（5000 tok smart_truncate）；reload 的历史**全量入窗**，仅靠 `fit` 逼近窗口时才压。 | 🟡 **不一致** |
| **大负载** | 2MB 工具输出直接存进会话 JSON 行；每次读历史都全量拉。 | 🔴 **无分表，热读被拖** |
| **压缩策略** | 位置式（截断/丢组），非相关性；丢了只能整轮 reload 才回得来。 | 🔴 **无检索召回** |

→ 地基（durable 源真相 + 无状态重建）**已具备**，本方案不动存储架构、**不写文件**，只加三件：**offload→handle 占位、大负载 DB 分表、相关性召回索引**。

## 3. Goals

- **offload-not-truncate**：窗口驱逐一条 tool_result 时，不再丢字节，而是换成紧凑占位 `{tool_call_id, digest, handle, tokens}`；全量仍在 store。
- **可回取**：给模型 `read_handle`（精确回取某条）+ `recall`（按相关性拉回分块）两个工具。
- **大负载 DB 分表**：超阈值正文落**同库独立表** `tool_result_body`（非文件/对象存储），原 item 行只留 handle，避免历史热读被大字段拖累。
- **相关性召回索引**：tool_result 分块入会话级 ES 命名空间，压缩从"位置式"升级为"relevance-aware"。
- **统一新/历史路径**：无论新产生还是 reload，超阈值一律 offload，投影一致。
- 全程分布式安全、无 worker 本地盘、可断线重建。

## 4. Non-Goals（v1）

- **不做模型级 compact 重写**（改写整段历史成新叙述）——沿用现有 rolling summary 作 Tier-1；本方案聚焦 tool_result 的 offload/召回。
- **不改 OpenAI-Responses 线格式 / 存储 schema 的破坏性迁移**——handle 与占位以**新 item 字段**增量承载。
- **不做跨会话的全局知识沉淀**——召回索引是**会话级**、有 TTL；跨会话沉淀属 user-memory 范畴。
- **不写任何文件、不引入对象存储**——全量 tool_result 本就在 DB；大负载走 **DB 独立表**，非 NAS/OSS。零文件写、零 path jail、零 orphan GC。（二进制/图片类产物走既有 `result.files`/artifact 路径，不在本方案文本 offload 范围。）

## 5. Decisions

1. **源真相 = 主 DB 的 `function_call_output`（全量），窗口是可重建投影。** 已成立，继续坚持；所有压缩只作用于投影副本。
2. **三个阈值，都是 token/DB 维度，无文件。**
   - **窗口软档 `OFFLOAD_SOFT_TOKENS ≈ 4000`（~16KB 文本，≈现 `cap_tool_result`）**：一条 tool_result 超此档且不在"近 N 轮"内 → 窗口里换占位。**这是主阈值**。
   - **存储分表档 `OFFLOAD_BODY_BYTES ≈ 32KB`**：正文移 **DB 独立表 `tool_result_body`**，原 item 行只留 digest+handle，避免历史热读拖大字段；小结果仍内联。
   - **存储硬顶 `MAX_STORED_BODY ≈ 1MB`**：仅防病态输出（某工具吐 10MB）撑爆行——超出截断+标注，是全案唯一有损点。
3. **handle 指向 DB，不是文件路径。** `handle = store://conv/{cid}/tool/{call_id}`，`read_handle` = 对 `tool_result_body`（或内联行）的一次**定向 DB 查询**。解析强校验 `cid/uid ∈ 当前 scope`（防越权），跨节点可解析（DB 是共享的）。
4. **召回索引会话级 + TTL。** 复用 `KnowledgeService`/ES，命名空间 `conv:{conversation_id}`；`delete_conversation` 挂清理钩子。写入**异步**（tool dispatch 后 upsert，不阻塞主循环）。
5. **压缩分级 relevance-aware。** `fit_to_budget` 的 L1/L2 从"位置式截断/摘要"升级为"驱逐→占位（可 `recall` 回取）"；被驱逐内容进召回索引。
6. **无状态、幂等。** offload 与占位写入和"持久化 tool 结果"同序；占位永远指向已落库的 handle（先写全量，后写占位/驱逐）。

## 6. Architecture

### 6.1 存储三层（都分布式，零本地盘）

| 层 | 存什么 | 存哪 | 现有基座 |
|---|---|---|---|
| **A 权威副本** | 每条 tool_result 全量 + 元数据（digest/tokens/sha） | 主 DB（`function_call_output` item），键 `conversation_id + call_id`；小结果正文内联 | ✅ `SqlStore` |
| **B 大负载分表** | 超 `OFFLOAD_BODY_BYTES` 的正文（纯文本） | 主 DB **独立表 `tool_result_body(cid, call_id, output_text)`**（同库不同表，避免热读拖大字段）；**无文件、无对象存储** | ✅ `SqlStore`（加一表） |
| **C 召回索引** | tool_result 分块 + 向量/BM25，命名空间 `conv:{cid}` | ES / 向量库，会话级 + TTL（索引写，非文件写） | ✅ `KnowledgeService` 混合检索 |

### 6.2 窗口投影四层（每步重算，不写回 store）

```
[稳定前缀: system/soul/tools]          ← 缓存友好，不变
[Tier-1 rolling summary]               ← 现有 summarizer（保留）
[Tier-2 最近 N 轮 tool_round 全量]      ← 高保真近场
[Tier-3 更早轮 → 占位 {digest,handle}]  ← offload，可 recall/read_handle 回取
```

### 6.3 offload 数据模型（占位 item 内容）

```jsonc
// function_call_output.content 增量字段（向后兼容：无这些字段=老全量行）
{
  "call_id": "call_abc",
  "output": "…前 512 char 摘要…",        // 占位时=digest；未 offload 时=全量
  "offloaded": true,                       // 新增：正文是否已从窗口驱逐
  "handle": "store://conv/{cid}/tool/call_abc",  // 始终指向 DB（内联行或 tool_result_body 表）
  "tokens": 18450,                         // 原始全量 token 估计
  "body_table": true,                      // 正文是否已挪到 tool_result_body 表（>OFFLOAD_BODY_BYTES）
  "indexed": true                          // 是否已入召回索引
}
```

### 6.4 数据流

```
工具执行 → 全量 result.content
  → [写] A层：append_items 持久化全量 function_call_output（源真相，永不删）
  → [同库] 若 > OFFLOAD_BODY_BYTES：正文移 B层 tool_result_body 表，A层行改存 handle+digest（纯 DB，无文件）
  → [异步] 分块 embed → C层 conv:{cid} 索引（indexed=true）
构造窗口(build_context / fit_to_budget)：
  → 近 N 轮：全量入窗
  → 更早轮：换占位 {digest, handle}
  → 模型需细节 → read_handle(handle) 精确回取 / recall(query) 相关性拉回
```

## 7. 工具 schema（新增两个只读工具）

```jsonc
// 精确回取被驱逐的某条结果（可分段，避免又爆窗口）
{
  "name": "read_handle",
  "description": "Re-fetch the full content of a tool result that was offloaded "
                 "from context (you'll see a placeholder with a `handle`). Use when "
                 "you need the exact details you saw earlier.",
  "parameters": {"type": "object", "properties": {
    "handle": {"type": "string"},
    "range":  {"type": "string", "description": "optional byte/line range, e.g. '0-4000'"}
  }, "required": ["handle"]}
}

// 按相关性把最相关分块拉回窗口（跨本会话所有已 offload 的 tool_result）
{
  "name": "recall",
  "description": "Search everything this conversation has already retrieved/read "
                 "(offloaded tool results) and pull back the chunks most relevant to "
                 "`query`. Use instead of re-running an expensive search.",
  "parameters": {"type": "object", "properties": {
    "query": {"type": "string"},
    "k": {"type": "integer", "default": 5}
  }, "required": ["query"]}
}
```

两者都在同一用户 scope 鉴权（复用 `ToolScope` + files serving 的 path jail），`handle` 的 `cid`/`uid` 必须与当前 scope 匹配，否则拒绝（防跨会话/跨用户越权）。

## 8. 新增/改动文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `agent/context_offload.py` | **新增** | offload 策略：`should_offload(result, tokens)`、`make_placeholder(...)`、`materialize(handle, range)`；handle 解析（纯 DB，无文件）。 |
| `agent/budgeting.py` | **改** | `fit_to_budget` 的 L1/L2 从"位置截断/摘要"改为"驱逐→占位"；`cap_tool_result` 统一走 offload（新旧路径一致）。 |
| `app/builder.py` | **改** | `items_to_messages` 识别占位字段，历史超阈值也走 offload（消除新/历史不对称）；`build_context` 组装四层窗口。 |
| `agent/tools/builtin/recall.py` | **新增** | `make_read_handle_tool(store)` / `make_recall_tool(index)`；注入式，不 import `app/`。 |
| `app/context_index.py` | **新增** | 会话级召回索引的 upsert/query/清理，封装 `KnowledgeService`（命名空间 `conv:{cid}`）。 |
| `api/protocol/responses_serializer.py` | **改** | `on_tool_result` 写入 offload 元数据字段（向后兼容）。 |
| `app/store/base.py` + `sql.py` | **改** | 新表 `tool_result_body(cid, call_id, output_text)`；`get_tool_result(cid, call_id)`（内联行或分表）；`function_call_output` content 增量字段（JSON，无 schema 破坏）。**无文件层。** |
| `app/routes/conversations.py` | **改** | `delete_conversation` 清理 C 层索引 + B 层 `tool_result_body` 行。 |
| `tests/test_context_offload.py` 等 | **新增** | offload/占位/回取/召回/越权拒绝/无状态重建/分表阈值。 |

## 9. 分布式 / 一致性要点

- **源真相全在 DB，worker 无状态、零本地盘、零文件**：按 `conversation_id` 重建窗口 → 水平扩展、断线恢复天然成立。
- **顺序**：先持久化全量（A 层）→ 再同库分表（B 层）/异步索引（C 层）；占位永远指向已落库 handle。异步失败可重试，不影响正确性（占位缺失时回退全量入窗）。
- **热会话可加 Redis read-through cache**：仅缓存 handle→正文，cache ≠ truth。
- **索引生命周期**：`conv:{cid}` 挂 TTL + `delete_conversation` 钩子，防膨胀；B 层 `tool_result_body` 行同删（外键级联 or 显式）。
- **鉴权**：`handle` 解析强校验 `uid/cid` ∈ 当前 scope；跨会话 `recall` 只搜本 `cid` 命名空间。
- **缓存友好**：offload 让稳定前缀 + 近场高保真更稳定，直接抬升 KV-cache 命中（成本主杠杆）。

## 10. 排期

1. **offload-not-truncate + `read_handle`**（最大可靠性收益，直接改善缓存）：`context_offload.py` + `budgeting` L1 改造 + serializer 元数据 + `read_handle` 工具 + store `get_tool_result`。
2. **大负载 DB 分表**：`tool_result_body` 表 + `OFFLOAD_BODY_BYTES` 分表 + `MAX_STORED_BODY` 硬顶 + delete 级联。**无文件写。**
3. **相关性召回**：`context_index.py`（会话级 ES）+ 异步 upsert + `recall` 工具 + `fit` L2 relevance-aware。
4. **统一历史路径 + 观测**：`items_to_messages` offload 对称化；offload/recall 命中率打点。
5.（后续）与子 agent 联动：`explore` 子 agent 的中间检索直接进召回索引，父 agent `recall` 复用。

## 11. Risks

- **占位过度导致模型频繁 recall**：靠"近 N 轮全量 + digest 足够自足"平衡；digest 里保留结论性首尾。
- **异步索引滞后**：`recall` 命中不到最新一轮属正常（最新一轮本就在窗口内全量）。
- **分表一致性**：正文写 `tool_result_body` 与 A 层行改 handle 在**同库事务**内完成（同库分表天然可事务，比跨系统 blob 简单）；无孤儿、无 GC。
- **DB 大字段膨胀**：`MAX_STORED_BODY≈1MB` 硬顶 + 分表隔离热读；病态超大输出截断+标注（唯一有损点）。
- **迁移**：老会话行无 offload 字段 → 读路径按"无字段=全量"兼容，无需回填。
- **成本**：embedding 召回索引增算力；仅对超软档的结果建索引，小结果不进。

## 12. 实现状态（2026-07-13）

**排期第 1 步已落地（offload-not-truncate + `read_handle`），并通过全量测试（644 passed）。** 第 2、3 步为纯增量层，未改第 1 步契约，留作后续。

**已落地：**
- `agent/context_offload.py`（新）：`OFFLOAD_SOFT_TOKENS`/`OFFLOAD_DIGEST_CHARS` 常量、`handle_for`/`parse_handle`、`make_placeholder`/`is_placeholder`。占位首行哨兵 `[offloaded tool result]`，含 handle + head digest，幂等（不会二次 offload）。
- `agent/budgeting.py`：`_truncate_tool_results_in_group` 由**有损截断**改为**offload**——把完整正文写入本轮 `run_bodies`，窗口只留占位。`fit` 的 L1 天然复用（对非保护 tool_round 生效）。cap_tool_result（当轮新结果的 head+tail）保持不变，友好且非 store-有损。
- `agent/agent.py` + `agent/tools/scope.py`：**放弃 ContextVar 方案**（异步生成器跨 task/context 驱动会触发 `reset(token)` "created in a different Context" —— 后台流式/resume 路径实测报错）。改为 `Agent.run` 每轮建一个 `run_bodies` dict，显式挂到 `self.budget.run_bodies` 与 `ToolScope.run_bodies`。每 run 一个新 Agent/新 dict → 子 agent 嵌套天然隔离。
- `agent/tools/builtin/read_handle.py`（新）：先查 `scope.run_bodies`（本轮未持久化的 offload），再查 store（历史轮），按 `scope.conversation_id` 鉴权（跨会话读不到）。支持 `start`/`count` 行范围。经 `_CONTEXT_RECOVERY_TOOLS` force-include，绕过 include/exclude。
- `app/store/{base,sql,memory}.py`：新增 `get_tool_result(cid, call_id)` —— 从既有 `function_call_output` item 取全量 `output`（**当前设计 store 已全量无损，无需分表即可跨 run 恢复**）。
- host 接线：`app/context_tools.py`（新）`wire_context_tools(state)`，在 `deps.py` / `lean_main.py` 与 `wire_subagents` 并列注册，绑定 `state.store`。
- 测试：`backend/tests/test_context_offload.py`（8 例）—— offload 可逆、幂等、小结果不动、`fit` 端到端、`read_handle` 命中 run_bodies / 命中 store / 会话隔离 / 空 handle。

**与原设计的偏差：**
- **§10-1 的 serializer 元数据未实现**：占位只活在窗口投影里（每步重算），store 始终写全量。`read_handle` 的历史恢复直接扫 `function_call_output`，不需要在持久化 item 上加 offload 字段。§6.3 的 offload 数据模型（持久化占位）留到需要"重载即占位"时再做（§10-4）。
- **in-run 恢复通道**由 ContextVar 改为显式 `run_bodies` dict（见上，鲁棒性原因）。

**待办（未动第 1 步契约）：**
- §10-2 `tool_result_body` 分表：纯热读优化（避免 `get_conversation_items` 拉大 blob）。`get_tool_result` 已工作，分表可后加。
- §10-3 相关性召回（`recall` + 会话级 ES）：较重，单独一块。
- §10-4 `items_to_messages` 重载路径对称化 + 命中率打点。
