# spawn_subagent —— 子 agent 与上下文防火墙 · 实现方案

**Date:** 2026-07-12
**Status:** ✅ Implemented (v1，后端全绿) · 单工具 + 主循环并发 dispatch（§15 修订，已删除 `spawn_subagents`）
**Branch:** `personal/yfei/agent-core`
**Builds on:** `agent/agent.py`(`Agent`/`AgentContext`)、`agent/tools/base.py`(`Tool`/`ToolBox`/`ToolScope`)、`app/builder.py`(`build_context`)、`app/deps.py`(`AppState`/`make_agent`)、`app/agent_config.py`(`AgentProfile`)、`app/providers.py`(`ProviderRouter`)。
**参考:** `docs/design/harness-gap-analysis.md` §5-①（此为该项的落地设计）。

---

## 1. Problem

PAI-Loop 当前**没有通用子 agent 机制**。`Agent.run` 是单请求的扁平工具循环，长任务/深度检索只能把所有中间产物（读 20+ 文档、多轮搜索）堆进主上下文，直接触发 "context rot"（输入越长模型越退化），并挤掉真正要推理的预算。

前沿的统一解法是 **context firewall**：让子 agent 在**独立上下文窗口**里做脏活，只回传 1–2K token 的摘要给主 agent。这既是上下文管理的核心手段，也是验证 agent（generator/evaluator 分离）、并行研究等能力的地基。

**关键约束（来自平台定位）**：PAI-Loop 是多租户、可治理的服务端平台。因此**不采用"主 agent 运行时自造 system prompt + 自行授予工具"的动态生成模式**（治理黑洞）——子 agent 的**能力封套必须是预先声明、可评审、可授权的**（复用 Agent Studio 的 `AgentProfile`），**任务由主 agent 运行时生成**。

## 2. Goals

- 一个 `spawn_subagent(agent_id, task)` 工具：主 agent 选一个**已配置 agent** 作为封套，传入运行时任务，在**隔离上下文**里跑完，只回传其最终摘要。
- 一个内置 `explore` 泛化 worker（只读检索），作为最高频的 context-firewall 用例，无需运营方预先编排。
- 复用现有无状态 `Agent` + `AgentContext`，**不改主循环**（v1）。
- 权限、并发、深度、超时全部有界；子 agent 失败不打断父循环。

## 3. Non-Goals（v1）

- **不做动态生成子 agent（模式 C）** —— 不允许主 agent 传入任意 system prompt/工具授予。
- ~~**不改 `Agent.run` 主循环**~~ —— **已修订**:为消除双胞胎工具,主循环改为并发 dispatch 同 turn 内的多个无依赖 tool call（`_dispatch_parallel`,保持消息/事件为模型原始调用顺序）。单发/`return_direct` 路径不变。SSE 线格式仍不变。见 §15。
- **不做子 agent 事件的实时透传**（子 agent 内部 TextDelta 不流到前端）—— v1 子 agent 是黑盒，只回传摘要；实时透传见 §9 后续项（需把 `dispatch` 改成可流式，单独排期）。
- **不做嵌套子 agent** —— 深度上限 1（子 agent 的 toolbox 不含 spawn 工具）。
- **不做跨请求的子 agent 会话持久化** —— 子 agent 是父 run 内的一次性调用。

## 4. Decisions

1. **子 agent = 用 `AgentProfile` 作封套 + 全新空 `AgentContext` 再跑一次 `Agent.run`。** 封套预定义、可治理；任务运行时生成。
2. **注入式 runner，避免分层反向依赖。** `agent/tools/builtin/spawn_subagent.py` 只定义 `Tool`，其 `fn` 调用一个注入的 `SubagentRunner`（住在 `app/`，持有 router/store/registry/agent_config）。这与 `knowledge_service`、`search_provider`、`make_sandbox_provider` 的注入方式一致。
3. **scope 继承走 contextvar。** 工具 `fn` 用 `get_current_tool_scope()` 读当前 `ToolScope`，把 `user_id / conversation_id / metadata` 传给子 run —— 子 agent 在**同一用户 scope**下运行，沙箱/KB 权限自动正确。子 agent 的 tool 批次再从子 ctx 构造**自己的** `ToolScope`（`agent_id`=子 id、`skill_mounts`=子的），沙箱 `_scope_key` 因 `agent:{agent_id}` 不同而拿到**独立会话**（隔离），但同一用户 NAS 目录。
4. **内置 `explore` worker 用保留 id。** runner 遇到 `agent_id=="explore"` 时合成一个只读 `AgentProfile`（不入库），避免运营方为最常见场景做配置。
5. **子上下文干净：空 history、不注入 memory/summary。** 纯 task-in / summary-out —— 这是 firewall 的意义。因此**不复用 `build_context`**（它会 resolve 历史/记忆/摘要），而是新写一个精简的 `build_subagent_context`，复用其中的**部件**（`render_stable_system_prompt`、`_select_tool_names`、`registry.build_toolbox`、`resolve_skill_mounts`）。
6. **并行 fan-out 由主循环并发 dispatch 承担，不做批量工具。**（**2026-07-12 修订，见 §15**）最初为绕开"主循环串行 dispatch"提供过 `spawn_subagents(tasks=[...])` 批量工具，但两个近义工具会让模型选择吃力、且可能意识不到该批量而退化成循环单发。改为对齐 Claude Code 的做法:**主循环把同一 turn 内多个无依赖的 tool call 并发 dispatch**,只保留单个 `spawn_subagent`,并行由模型在一个 turn 里发多个调用自然获得。

## 5. Architecture

### 5.1 新增/改动文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `app/subagent.py` | **新增** | `SubagentRunner`：查 profile → 建子 ctx → 跑子 `Agent.run` → 收敛为 `SubagentResult`。 |
| `agent/tools/builtin/spawn_subagent.py` | **新增** | `make_spawn_subagent_tool(runner)` / `make_spawn_subagents_tool(runner)`：定义 `Tool`，`fn` 调 runner。 |
| `agent/soul.py` | **新增函数** | `render_subagent_system_prompt(profile, tool_names)`：子 agent 的稳定系统提示（含"只回传自足摘要"纪律）。 |
| `app/builder.py` | **新增函数** | `build_subagent_context(...)`：空 history 的精简 ctx 构造（复用现有部件）。 |
| `app/deps.py` | **改** | 组装 `AppState` 后创建 `SubagentRunner(state)` 并把 spawn 工具注册进 `registry`（后置注册，见 §5.4）。 |
| `app/agent_config.py` | **改** | `CapabilityConfig` 增加 `subagent` 能力项（默认 enabled 门控）；可选 `AgentProfile.settings["subagent"]` 白名单。 |
| `tests/agent/test_subagent.py` | **新增** | `FakeLLM` 脚本化父→子链路 + runner 单元测试。 |

### 5.2 `SubagentRunner`（`app/subagent.py`）

```python
@dataclass
class SubagentResult:
    ok: bool
    summary: str                 # 子 agent 最终文本（回传给父模型）
    usage: Usage                 # 子 run 累计用量（用于计费聚合）
    finish_reason: str           # stop / max_steps / error / timeout
    error: Optional[str] = None

class SubagentRunner:
    def __init__(self, state: "AppState"):   # 持有 router/store/registry/agent_config
        self._state = state

    async def run(self, *, agent_id: str, task: str, scope: ToolScope,
                  depth: int = 0) -> SubagentResult:
        # 1) 治理：深度上限
        if depth >= MAX_SUBAGENT_DEPTH:            # =1，禁止嵌套
            return SubagentResult(False, "", Usage(), "error",
                                  error="subagent nesting not allowed")
        # 2) 解析封套
        profile = self._resolve_profile(agent_id) # 库内 profile 或合成 explore worker
        if profile is None:
            return SubagentResult(False, "", Usage(), "error",
                                  error=f"unknown agent_id: {agent_id}")
        # 3) 选模型（子 profile 的 model 或继承默认）
        model_id = profile.model or self._state.default_model
        llm = self._state.router.get_llm(model_id)
        cfg = self._state.router.get_config(model_id)
        # 4) 建"干净"子上下文：空 history、无 memory/summary、
        #    继承同一用户 scope，工具集来自子 profile（且剔除 spawn_*）
        child_ctx = build_subagent_context(
            profile=profile, task=task, parent_scope=scope,
            registry=self._state.registry, depth=depth + 1,
        )
        # 5) 跑子 loop，把 TextDelta 收敛成 summary，累计 usage
        agent = self._state.make_agent(
            llm=llm, context_window=cfg.context_window,
            max_output_tokens=cfg.max_output_tokens)
        child_agent = _with_max_steps(agent, profile)  # 子步数上限更小(默认10)
        return await _collect_run(child_agent, child_ctx,
                                  timeout=SUBAGENT_TIMEOUT_SECONDS)
```

`_collect_run` 迭代 `agent.run(child_ctx)` 事件：累加 `TextDelta.text` → `summary`；取末个 `RunCompleted.usage`；`RunFailed` → `ok=False`；整体裹 `asyncio.wait_for(..., SUBAGENT_TIMEOUT_SECONDS)`。**任何异常都转成 `SubagentResult(ok=False, error=...)`，绝不抛进父循环**（与 `web_fetch` 的"从不 raise"约定一致）。

`_resolve_profile`：
- `agent_id == "explore"` → 合成只读 worker（不入库）：
  ```python
  AgentProfile(id="explore", name="Explore",
      instructions=_EXPLORE_INSTRUCTIONS,       # "深挖并只回传自足的紧凑摘要"
      tools=AgentToolsConfig(include=[
          "knowledge_search","view_file","grep_file","web_fetch","web_search"]),
      settings={"max_steps": 8})
  ```
- 否则 `next((a for a in state.agent_config.agents if a.id == agent_id), None)`（复用 `builder._resolve_agent_profile` 的查法，`builder.py:430`）。

### 5.3 `build_subagent_context`（`app/builder.py`）

复用现成部件，但 **history 恒为空**、**不注入 memory/summary/instructions 的 volatile 块**：

```python
def build_subagent_context(*, profile, task, parent_scope, registry, depth):
    tool_names = _select_tool_names(registry, profile, force=_skill_force(profile))
    tool_names = [n for n in tool_names if n not in _SPAWN_TOOL_NAMES]  # 断递归
    toolbox = registry.build_toolbox(tool_names)
    system_prompt = render_subagent_system_prompt(profile, tool_names=tool_names)
    md = dict(parent_scope.metadata)              # 继承 sandbox_env/aliyun/admin
    md["default_kb_ids"] = profile.knowledge.kb_ids  # 子 agent 自己的 KB 软默认
    md["subagent_depth"] = depth
    return AgentContext(
        system_prompt=system_prompt,
        history=[],                               # ★ firewall：空历史
        current_turn=Message("user", task),
        attachments=[], hints=[],
        tools=toolbox, run_vars=RunVars(),
        context_block="",                         # ★ 不带 memory/summary
        user_id=parent_scope.user_id,             # ★ 同一用户 scope
        conversation_id=parent_scope.conversation_id,
        metadata=md,
        agent_id=profile.id,                      # ★ 子 id → 沙箱独立会话
        skill_mounts=resolve_skill_mounts(profile, ...),
        skill_fingerprint=skill_mount_fingerprint(...),
    )
```

> 子 run 内部，`Agent.run` 会用 `child_ctx` 的字段自建每批 `ToolScope`（`agent.py:256`），因此子 agent 的 sandbox `_scope_key` 天然带 `agent:{child_id}` → 与父隔离但同用户目录（`sandbox_providers.py:149` 派生逻辑）。

### 5.4 装配（`app/deps.py`）—— 破环

`SubagentRunner` 需要 `registry`，而 `registry` 由 `build_default_registry` 产出——存在循环。解法：**后置注册**（runner 只在**请求时**被调用，此时 `AppState` 已完整）：

```python
# rebuild_app_state_from_config(...) 里，registry/state 组装完成后：
if _subagent_enabled(agent_config):              # capability 门控 + agents>1 或 explore 开
    runner = SubagentRunner(state)
    registry.register(make_spawn_subagent_tool(runner))
    registry.register(make_spawn_subagents_tool(runner))  # 并行批量版
```

`registry.register(...)` 沿用 `build_default_registry` 注册工具的同一机制（`defaults.py`）。这样**不改 `build_default_registry` 签名**、无循环依赖。工具随 config 热重载一并重建（`rebuild_app_state_from_config`，`deps.py:59`）。

### 5.5 工具 schema

```python
# spawn_subagent —— 命名封套 + 运行时任务
{
  "name": "spawn_subagent",
  "description": "把一个自足的子任务委派给一个已配置的子 agent，在隔离上下文里执行，"
                 "只返回其最终摘要。用于需要深度检索/多步探索、但不该污染主对话的活。"
                 "agent_id 传 'explore' 使用内置只读检索 worker。",
  "parameters": {
    "type": "object",
    "properties": {
      "agent_id": {"type": "string",
                   "description": "子 agent 的 id（见可用 agent 列表），或 'explore'"},
      "task": {"type": "string",
               "description": "给子 agent 的完整、自足的任务描述（它看不到本对话历史）"}
    },
    "required": ["agent_id", "task"]
  }
}
```

`fn` 实现（`spawn_subagent.py`）：

```python
def make_spawn_subagent_tool(runner: SubagentRunner) -> Tool:
    async def fn(agent_id: str, task: str) -> str:
        scope = get_current_tool_scope()
        depth = int(scope.metadata.get("subagent_depth", 0))
        res = await runner.run(agent_id=agent_id, task=task, scope=scope, depth=depth)
        if not res.ok:
            return f"[subagent {agent_id} 失败] {res.error}"
        logger.info("[subagent] %s usage=%s finish=%s", agent_id,
                    res.usage.total, res.finish_reason)
        return res.summary                    # → 成为父侧 ToolResult.content
    return Tool(name="spawn_subagent", description=..., parameters=..., fn=fn,
                permission="auto")
```

`spawn_subagents(tasks: [{agent_id, task}])` 版：`fn` 内 `asyncio.gather(*[runner.run(...) for t in tasks])`（受 `SUBAGENT_MAX_CONCURRENCY` 信号量约束），把各摘要拼成带小标题的一段返回。

## 6. Data flow

```
主 agent 发起 spawn_subagent 工具调用
  → ToolBox.dispatch 设 ToolScope contextvar（含 user/conv/metadata）
    → fn 读 scope → SubagentRunner.run(agent_id, task, scope, depth)
        → 解析 AgentProfile（库内 or explore 合成）
        → build_subagent_context：空 history + 子 system_prompt + 子 toolbox(去 spawn)
        → 子 Agent.run(child_ctx)：独立窗口跑完整工具循环
        → 收敛 TextDelta → summary，累计 usage，wait_for 超时保护
    → 返回 summary 字符串
  → 父侧 append 为 tool message（经 budget.cap_tool_result 截断）
  → 父 agent 基于摘要继续推理
```

**Firewall 生效点**：子 agent 读的 20+ 文档、多轮搜索结果全在**子窗口**，父窗口只多了一条 ≤cap 的摘要。

## 7. 治理 / 安全边界

| 维度 | 措施 |
|---|---|
| **能力门控** | `CapabilityConfig(id="subagent")` 未 enabled → 不注册工具（与其它能力一致）。 |
| **深度** | `MAX_SUBAGENT_DEPTH=1`：子 ctx 剔除 spawn_* 工具 + `subagent_depth` 计数双保险。 |
| **并发** | `SUBAGENT_MAX_CONCURRENCY`（进程级信号量）+ 每父 run 子调用数上限。 |
| **步数/时长** | 子 `max_steps`（默认 8–10，可 `profile.settings["max_steps"]`）+ `SUBAGENT_TIMEOUT_SECONDS` 硬超时。 |
| **RBAC** | v1：任何**已配置** agent id 可 spawn（与 `GET /v1/agents` 全员可见一致）；`explore` 始终可用。**可选白名单** `profile.settings["subagent"]["allow"]` 限制某 agent 能 spawn 谁。未来接 per-user agent ACL。 |
| **无动态封套** | 只接受 `agent_id`，**不接受任意 instructions/工具授予** —— 杜绝模式 C 的越权。 |
| **失败隔离** | runner 内全异常收敛为错误字符串，父循环不受影响。 |
| **凭据** | 子 run 继承同一 `metadata`（含 `aliyun_sandbox_env`），但沙箱按 `agent_id` 独立会话 —— 不跨子 agent 泄漏运行时状态。 |

## 8. 计费 / 可观测（已知边界）

- 子 usage **不计入**父 `RunCompleted.usage`（父只统计自身流）。runner 返回 `res.usage`，v1 落 **日志**；建议同时写入 `scope.metadata["subagent_usage"]` 累加，由 route/`RunManager` 在 run 结束时汇总入账（后续项）。
- v1 子 agent 对前端**不可见**（黑盒），仅父侧 `ToolStarted/ToolCompleted/ToolResult(output=摘要)` 可见。UI 可据工具名 `spawn_subagent` 特殊渲染为"委派卡片"。

## 9. 后续项（非 v1）

1. **子事件实时透传**：把 `ToolBox.dispatch` 从"await 返回"改为"async generator 产出事件"，让子 `AgentEvent` 以 `subagent` 信封透传到 SSE（新增 `response.subagent.*` 帧）。需动主循环，单独 spec。
2. **验证 agent**：基于本机制的 generator/evaluator 分离——一个只读子 agent 评审主产物（对应 gap-analysis §5-④）。
3. **usage 入账聚合** + 子 run 的 OTel span 嵌套。
4. **per-user agent ACL**：spawn 时按用户可见 agent 校验 `agent_id`。

## 10. Testing

- **`FakeLLM` 脚本化端到端**（复用 agent-loop-refactor 引入的 FakeLLM 模式）：父 FakeLLM 先吐一个 `spawn_subagent` 工具调用 → runner 用子 FakeLLM（脚本化返回一段摘要）→ 断言父侧收到 `ToolResult(output=摘要)` 并据此收尾。无网络、确定性。
- **`build_subagent_context` 单测**：history 为空、`context_block==""`、toolbox **不含** `spawn_subagent`、`user_id/conversation_id` 继承自 parent_scope、`agent_id==profile.id`、KB 软默认为子 profile 的 `kb_ids`。
- **runner 单测**：未知 `agent_id` → `ok=False`；`depth>=1` → 拒绝嵌套；超时 → `finish_reason="timeout"`；子 `RunFailed` → 错误字符串；`explore` 合成 profile 只读工具集正确。
- **并发**：`spawn_subagents` 受信号量约束、聚合多摘要。
- **门控**：capability 关 → 工具未注册；`agents<=1` 且 explore 关 → 不注册。
- 现有 import-lean / boot 隔离 gate 保持绿。

## 11. Sequencing

1. `render_subagent_system_prompt` + `build_subagent_context` + 单测。
2. `SubagentRunner` + `_collect_run` + `explore` 合成 + 单测。
3. `spawn_subagent` 工具 + `deps.py` 后置注册 + 门控 + FakeLLM 端到端。
4. `spawn_subagents` 并行批量版 + 并发单测。
5.（后续 spec）子事件透传 / 验证 agent / usage 入账。

## 12. Risks

- **`registry.register` 后置注册的热重载**：确认 `rebuild_app_state_from_config` 每次重建 registry 时都会重跑后置注册（把注册逻辑放进该函数体内，而非只在首次 boot）。
- **子 agent 摘要过长**：靠子 system_prompt 的"紧凑自足"纪律 + 父侧 `cap_tool_result` 双重约束；必要时给子加一步显式 synthesis。
- **沙箱会话膨胀**：每个 (user, child_agent) 一个沙箱会话，`ScopedSandboxProvider` 的 idle 回收（`session_idle_seconds`）已覆盖；关注并发 fan-out 时的会话峰值。
- **成本**：子 agent 是独立 LLM run，fan-out 会放大 token（Anthropic 观察多 agent ~15x）。默认 `explore` 用较小 `max_steps`；并发信号量兜底。
- **provider 缺 key**：`router.get_llm` 对缺 `api_key_env` 抛 `RuntimeError` → 被 runner 收敛为错误字符串，不炸父循环。

---

## 13. 实现状态（2026-07-12）

后端 v1 全部落地并通过测试（`tests/test_subagent.py` 9 例 + `test_lean_import_isolation` + 全量 `635 passed`）。与本文设计的差异均为向前兼容的收敛。

### 13.1 后端落地清单

| 文件 | 状态 | 与设计的差异 |
|---|---|---|
| `app/subagent.py` | ✅ 新增 | `SubagentRunner` / `SubagentResult` / `_collect_run` / 内置 `explore` 合成 / `wire_subagents(state)` 后置注册 / `subagent_enabled()` 门控。`explore` 工具集扩为 `knowledge_search, view_file, grep_file, list_knowledge_bases, web_search, web_fetch, shell, code_interpreter, current_datetime`（与注册表取交集）。 |
| `agent/tools/builtin/spawn_subagent.py` | ✅ 新增 | `SPAWN_TOOL_NAMES`（=`("spawn_subagent",)`）/ `make_spawn_subagent_tool`。不 import `app/`。**§15 修订后删除了 `make_spawn_subagents_tool` 及 `SUBAGENT_MAX_CONCURRENCY`/`SUBAGENT_MAX_BATCH`** —— 并行改由主循环承担。 |
| `agent/soul.py` | ✅ 新增 | `render_subagent_system_prompt(...)` 追加 `_SUBAGENT_PROTOCOL`（"只回传自足摘要"纪律）。 |
| `app/builder.py` | ✅ 新增 | `build_subagent_context(...)`：空 history、`context_block=""`、剥离 `SPAWN_TOOL_NAMES`、继承父 scope、`agent_id=profile.id`、KB 软默认切子 profile。**未复用** `build_context`（避免碰 store/memory/summary）。 |
| `app/agent_config.py` | ✅ 改 | 默认 `main` agent 的 `tools.include` 加入 `spawn_subagent`（§15 修订后仅此一个）；新增 `CapabilityConfig(id="subagent", kind="core_tool", enabled=True, status="ready", permission="auto")`；`apply_runtime_status` 的 ready 分支纳入 `subagent`。 |
| `app/deps.py` + `app/lean_main.py` | ✅ 改 | 破环：`wire_subagents(state)` 在 boot（lean_main，直接建 registry）与每次热重载（deps 的 `rebuild_app_state_from_config`）**两处**都调。 |
| `tests/test_subagent.py` | ✅ 新增 | 干净子 ctx / explore happy path / 未知 id / 拒绝嵌套 / 空 task / 两个 spawn 工具（含并行聚合与空、超额拒绝）。 |

### 13.2 治理默认值（已固化）

- 深度上限 `MAX_SUBAGENT_DEPTH=1`（子 ctx 去 spawn 工具 + `metadata["subagent_depth"]` 计数双保险）。
- `SUBAGENT_TIMEOUT_SECONDS=240`、`SUBAGENT_MAX_STEPS=12`（可 env 覆盖）。并发 fan-out 由主循环 `TOOL_DISPATCH_CONCURRENCY=8` 信号量兜底（§15）。
- capability `subagent` **缺省即开**（`subagent_enabled` 对旧 config 返回 True）；个体 agent 仍需 `tools.include` 显式纳入才拿到工具。

## 14. 前端如何"生效"

**结论：能力卡片自动出现，但工具开关需接一处映射（3 行）。** 现状分两层：

1. **能力卡片自动渲染** —— `frontend/src/components/SettingsView.tsx` 的 `systemTools(doc)` 遍历 `doc.capabilities` 中 `kind==="core_tool"` 且非 control-plane 的项动态生成工具清单。后端已随 config 下发 `subagent` 能力，故 Settings 的 Agents 页会**自动**多出一张 "Subagents"（`cap.name`）卡片、状态 Ready，**无需改前端**即可显示与在能力总览里读到。

2. **开关写入需一处 `TOOL_BUNDLES` 映射（✅ 已接）** —— 关键错配：能力 **id 是 `subagent`**，而真正注册的工具名是 **`spawn_subagent`**。UI 勾选时 `toggleAgentTool` 通过 `toolBundle(id)` 把工具名写进 `agent.tools.include`。若无映射，勾选会把字面量 `"subagent"` 写进 include —— 匹配不到注册工具，agent 拿不到 spawn 工具；`isToolEnabled` 也会把已默认带工具的 `main` 显示成"未开"。这与 `sandbox → code_sandbox → [code_interpreter, shell, publish_artifact]` 是同一模式。已补于 `SettingsView.tsx`：

   ```ts
   const TOOL_BUNDLES: Record<string, string[]> = {
     knowledge_search: ["knowledge_search"],
     code_sandbox: ["code_interpreter", "shell", "publish_artifact"],
     subagent: ["spawn_subagent"],   // ← 已接（§15 修订:单工具,故单名）
   };
   ```

   勾选 "Subagents" → 写入 `spawn_subagent`；`main` 因默认已含该工具而正确显示为已开。

3. **默认 agent 无需任何前端操作** —— `main` 的 `tools.include` 已在后端硬编码含 `spawn_subagent`，开箱即用；上面的前端改动只影响**其它/新建 agent** 的可视化勾选与开关状态一致性。

> 保存与热重载链路已就绪：Settings 的保存走 `PUT /v1/config` → `rebuild_app_state_from_config` → `wire_subagents` 一并重建注册，无需重启。

## 15. 修订：删除 `spawn_subagents`，并行下沉到主循环（2026-07-12）

**动机。** 单发 `spawn_subagent` 与批量 `spawn_subagents` 两个近义工具是 tool-design 反模式：描述高度相似 → 模型选择分裂,且可能意识不到该批量而退化成"循环单发"（恰好丢掉并行）。顶尖 harness（Claude Code 的 `Task`、OpenHands 的 delegate、LangGraph 的 handoff）都**只有一个委派原语**;Claude Code 的并行来自**主循环把一个 turn 内多个 tool_use 并发 dispatch**,而非第二个工具（其系统提示明确引导"send multiple tool uses in a single message so they run concurrently"）。

**改动。**

| 文件 | 改动 |
|---|---|
| `agent/agent.py` | 新增 `TOOL_DISPATCH_CONCURRENCY=8` 与 `_dispatch_parallel(tools, pairs, scope)`。主循环:当一个 turn 有 **>1 个 tool call 且无 `return_direct`** 时,先 `asyncio.gather`(受信号量约束)并发跑完,再按**原始调用顺序**回填 message/事件;单发 / `return_direct` 仍走原 await-per-call 路径(**字节级不变**)。 |
| `agent/tools/builtin/spawn_subagent.py` | 删 `make_spawn_subagents_tool` 及 `_MAX_BATCH`/`_MAX_CONCURRENCY`/`_sem`;`SPAWN_TOOL_NAMES=("spawn_subagent",)`;单工具描述新增"要 fan-out 就在一个 turn 里发多个 spawn_subagent 调用"。 |
| `app/subagent.py` | `wire_subagents` 只注册单工具。 |
| `app/agent_config.py` | `main.tools.include` 去掉 `spawn_subagents`。 |
| `frontend/SettingsView.tsx` | `TOOL_BUNDLES.subagent = ["spawn_subagent"]`。 |
| `tests/test_agent_parallel_dispatch.py` | **新增**:并发探针证明同 turn 两个 tool call `peak==2`(重叠执行)、`ToolResult` 仍按 `c1,c2` 原序;单发 `peak==1`。 |
| `tests/test_subagent.py` | 删两个批量工具用例,加 `SPAWN_TOOL_NAMES` 单元断言。 |

**正确性要点。** ① 消息回填与事件产出严格按 `pairs` 原序 → LLM 历史 `function_call`/`function_call_output` 配对与顺序不变,replay-safe。② 并发安全:`ToolBox.dispatch` 用 contextvar 存 `ToolScope`,`asyncio.gather` 每个协程跑在各自 copied context 里,scope set/reset 不串（与并行子 agent 同一机制）。③ `dispatch` 从不 raise（失败即 `ToolResult(ok=False)`）,故无需 `return_exceptions`。④ `return_direct` 会短路整个 batch,必须保持串行,故被显式排除在并行路径外。

**回归面。** 仅**多工具且非 return_direct**的 turn 行为改变(现在并发+结果一次性回填);占绝大多数的单工具 turn 与 HITL/return_direct 路径逐字不变。全量 `pytest` 通过。
