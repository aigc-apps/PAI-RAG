# PAI-RAG 技术开发文档

一个**~2400 行 Python 的小型自治 Agent**，强调"骨架够清楚就能扩"：

- **核心 ~1100 行**（执行循环 + 7 个原子工具 + LLM 客户端 + 持久化 + 技能加载），坚持单一执行路径、无 SDK 抽象、无注册表
- **两个入口** 复用同一份核心，分别覆盖：Next.js Web、ACP（IDE 集成，Zed/VSCode）
- **四层自进化记忆系统**：L1 索引 → L2 事实 → L3 SOP → L4 原始归档；主回答交付后由后台 review agent 通过 `start_long_term_update` 异步沉淀经验
- **OpenAI 兼容多后端**：默认 Qwen via DashScope，可一行切换 OpenAI / DeepSeek / 本地 vLLM·Ollama / OpenRouter

本文档是**技术开发文档**，重点讲清楚每个模块为什么这样写、关键设计点在哪。使用、启动和 API 调试说明见 [README.md](./README.md)。

---

## 目录

- [整体架构](#整体架构)
- [一次任务的完整数据流](#一次任务的完整数据流)
- [核心模块详解](#核心模块详解)
  - [执行循环 agent_loop.py](#1-执行循环-agent_looppy)
  - [工具系统 tools.py](#2-工具系统-toolspy)
  - [LLM 客户端 llm_client.py](#3-llm-客户端-llm_clientpy)
  - [会话持久化 session_store.py](#4-会话持久化-session_storepy)
  - [技能动态加载 skill_manager.py](#5-技能动态加载-skill_managerpy)
- [两个入口](#两个入口)
  - [React `frontends/react`](#react-frontendsreact)
  - [ACP `frontends/acp`](#acp-frontendsacp)
- [记忆架构：四层自进化系统](#记忆架构四层自进化系统)
- [关键设计决策](#关键设计决策)
- [扩展指南](#扩展指南)
- [使用和启动](#使用和启动)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  入口层（二选一，复用同一份 core）                                      │
│   Next.js(frontends/react)              ACP(frontends/acp)          │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  agent_runner_loop()  ←  agent_loop.py                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ LLM Chat │───▶│ Parse        │───▶│ handler.dispatch()       │  │
│  │ (stream) │    │ tool_calls   │    │   do_code_run            │  │
│  │          │    │              │    │   do_file_read/write/.   │  │
│  │          │◀───│ tool_results │◀───│   do_ask_user            │  │
│  │          │    │ + next_prompt│    │   do_update_working_..   │  │
│  └────┬─────┘    └──────────────┘    │   do_start_long_term_..  │  │
│       ▼                              └──────────────────────────┘  │
│  LLMClient (openai SDK)              GenericHandler (tools.py)     │
│   history[] ◀── trim_history()         + skill 动态分发              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  memory/                                                            │
│   ├ global_index.txt   (L1, 注入 system prompt)                     │
│   ├ global_facts.txt   (L2, agent 按需 file_read)                   │
│   ├ *_sop.md           (L3, agent 按需 file_read/write)             │
│   ├ L4_raw_sessions/   (L4, 前端归档)                                 │
│   ├ users/<user_id>/   (普通用户私有长期记忆)                           │
│   ├ sessions_v2.sqlite3 (跨进程会话恢复)                               │
│   └ sessions/          (.gitkeep，占位目录)                            │
└─────────────────────────────────────────────────────────────────────┘
```

**模块职责一览：**

| 层 | 模块 | 职责 | 行数 |
|---|------|------|-----:|
| Core | `agent_loop.py` | 执行循环、工具调度、退出判断 | 123 |
| Core | `agent_events.py` | ACP 风格结构化事件构造 | - |
| Core | `tools.py` | 7 个原子工具实现 + GenericHandler | 414 |
| Core | `llm_client.py` | OpenAI 兼容流式调用、历史裁剪 | 167 |
| Core | `session_store.py` | SQLite 会话持久化 | 90 |
| Core | `skill_manager.py` | 技能动态加载 + 斜杠命令分发 | 140 |
| Frontend | `frontends/react/` | Next.js Web UI | - |
| Frontend | `frontends/acp/server.py` | ACP（JSON-RPC over stdio） | 533 |
| Frontend | `frontends/acp/jsonrpc.py` | 双向 JSON-RPC 框架 | 97 |

---

## 一次任务的完整数据流

```
用户输入 "帮我查看 EAS 服务状态"
         │
         ▼
┌─ 前端 ────────────────────────────────────────────────────────┐
│ 1. 构造 handler（继承上一任务的 history_info + working）        │
│ 2. 如有 prev_handler → 拼 _anchor_prompt() + 用户消息          │
│ 3. 调用 agent_runner_loop()                                    │
└────────────────────┬──────────────────────────────────────────┘
                     ▼
┌─ agent_loop.py ───────────────────────────────────────────────┐
│ for turn in 1..max_turns:                                      │
│   ① client.chat()  → 收集完整 Response                          │
│   ② 解析 response.tool_calls                                   │
│      └ 无工具调用 → return NO_TOOL_CALL（任务结束）              │
│   ③ 顺序执行: handler.dispatch(name, args, response)           │
│      └ 返回 StepOutcome(data, next_prompt, should_exit)        │
│   ④ 回填 OpenAI 风格 tool message                              │
│      [{role:'tool', tool_call_id, content}, ...]               │
│   ⑤ turn_end_callback → 提取 <summary>、注入轮数警告            │
│   ⑥ 拼装下轮 new_messages = tool_results + next_prompt         │
└────────────────────┬──────────────────────────────────────────┘
                     ▼
┌─ 前端（收尾）─────────────────────────────────────────────────┐
│ 4. archive_session → dump 到 L4_raw_sessions/                 │
│ 5. store.save → 持久化到 sessions_v2.sqlite3                   │
│ 6. prev_handler = handler（传递给下一任务）                     │
└───────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. 执行循环 `agent_loop.py`

整个 agent 的心脏，~120 行，可以一口气读完。

**StepOutcome —— 工具返回值的三元组：**

```python
@dataclass
class StepOutcome:
    data: Any  # 回填给 LLM 的 tool message content
    next_prompt: Optional[str]  # 注入下轮的文本提示（None = 任务结束）
    should_exit: bool  # 强制退出
```

`next_prompt` 是控制 agent 行为的**核心杠杆**——工具不仅返回数据，还能通过 `next_prompt` 向 LLM 注入额外指令（如工作记忆、SOP 文本、轮数警告）。这是 mini-agent 实现"工具引导 agent 行为"的核心机制。

**退出条件优先级：**

1. `should_exit=True` → 立即退出（EXITED）
2. `next_prompt=None` → 当前任务完成（CURRENT_TASK_DONE）
3. 无 tool_calls → 模型主动结束（NO_TOOL_CALL）
4. 超过 max_turns → 强制退出（MAX_TURNS_EXCEEDED）

**BaseHandler —— 约定式分发：**

```python
class BaseHandler:
    def dispatch(self, tool_name, args, response, index=0):
        method = getattr(self, f"do_{tool_name}", None)  # 约定: do_<tool_name>
        ...
```

不用注册表、不用装饰器，直接按命名约定分发。加新工具 = 加一个 `do_xxx` 方法，零配置开销。

**turn_end_callback —— 轮间钩子：**

每轮所有工具执行完后调用，职责：
- 提取 `<summary>` 标签作为历史摘要（如果 LLM 没写，则自动从 tool_call 生成兜底摘要）
- 每 7 轮注入 `[DANGER]` 警告，防止 agent 无效重试陷入死循环

### 2. 工具系统 `tools.py`

主链路暴露 7 个工具；后台 memory review 额外使用 `start_long_term_update`。

**执行类（与外部世界交互）：**

| 工具 | 关键设计点 |
|------|-----------|
| `code_run` | 流式输出（`threading.Thread` 读 stdout），超时 `proc.kill()`；从 LLM 回复的代码块自动抽代码；`cancel_evt` 可被前端打断（ACP `session/cancel` 用） |
| `file_read` | keyword 搜索时用滑动窗口（`collections.deque`）保留上下文；FileNotFoundError 时用 `difflib.SequenceMatcher` 做模糊路径推荐 |
| `file_patch` | **唯一性约束**：匹配 0 次或 >1 次都拒绝，迫使 LLM 先读再改 |
| `file_write` | 从 LLM 回复中提取 `<file_content>` 标签或代码块；支持 `{{file:path:start:end}}` 引用展开 |
| `ask_user` | Web/ACP 重写为 queue 阻塞，前端把答案 put 进来唤醒 |

**元认知类（agent 管理自身状态）：**

| 工具 | 关键设计点 |
|------|-----------|
| `update_working_checkpoint` | 写入 `handler.working`，每轮通过 `_anchor_prompt()` 自动注入到下轮 prompt |

**后台 review 专用：**

| 工具 | 关键设计点 |
|------|-----------|
| `start_long_term_update` | 不暴露给主链路；仅提供给后台 memory review agent。读取 `memory_management_sop.md` 全文作为 next_prompt，引导 review agent 按决策树分类信息并更新 L1/L2/L3 |

**`_anchor_prompt()` —— 工作记忆注入：**

```
### [WORKING MEMORY]
<history>
[Agent] 执行了 list-services，发现 3 个运行中
[Agent] 读取了 EAS SOP，提取关键命令
</history>
Current turn: 5
<key_info>用户需要检查 cn-hangzhou 区域的服务状态...</key_info>
有不清晰的地方请再次读取 memory/aliyun_eas_sop.md
```

每轮拼到 tool_result 后面，让 LLM 始终保持对任务进度和关键信息的感知。`skip` 参数避免同一轮多个工具重复注入。

### 3. LLM 客户端 `llm_client.py`

**~170 行，基于 `openai` 官方 Python SDK。**

支持任意 OpenAI 兼容端点：DashScope (Qwen) / OpenAI / DeepSeek / vLLM / Ollama / OpenRouter / Azure。鉴权统一走 `Authorization: Bearer <API_KEY>`，由 SDK 处理。

**关键设计点：**

**a) `chat()` 是双重返回的 generator：**

```python
def chat(self, system, new_messages, tools):
    """yields text chunks during streaming, returns Response."""
    ...
    for chunk in stream:
        if delta.content:
            yield delta.content  # 实时打印用
    return Response(content=..., tool_calls=..., stop_reason=...)
```

调用方通过 `try/except StopIteration as e: response = e.value` 同时拿到流式文本和结构化结果。

**b) Tool call 流式累积：**

OpenAI 流式响应里 `tool_calls` 按 `index` 分片到达，每一片可能只含 `id` / `name` / `arguments` 中的一个或几个字符：

```python
for tc in tcs:
    idx = getattr(tc, "index", 0) or 0
    slot = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
    if tc.id:
        slot["id"] = tc.id
    if fn.name:
        slot["name"] = fn.name
    if fn.arguments:
        slot["args"] += fn.arguments  # 字符串拼接
```

最后一次性 `json.loads(slot['args'])` 还原参数 dict。

**c) History 裁剪 `trim_history`：**

按 `len(text) // 4` 粗估 token 数，超过阈值时从最早消息开始删，**且跳过 `assistant`/`tool` 消息**——避免出现"孤立 tool message"导致下次请求 400。

**d) Prompt cache：**

服务端自动处理（DashScope / DeepSeek / OpenAI 都做前缀缓存），客户端**零配置**。

### 4. 会话持久化 `session_store.py`

- **事务写入**：会话快照写入 `memory/sessions_v2.sqlite3`，SQLite 负责事务一致性
- **单用户会话**：HTTP 后端默认使用服务用户 `__server__`，列表/读取/删除/取消/继续对话均按 `session_id` 操作；历史其他 `user_id` 记录默认不展示
- **运行状态**：持久化 `status`、`active_run_id`、`workspace_path`；同一 session 在 `running` 时拒绝新 prompt，`waiting_user` 时才允许把下一条消息作为 ask_user 回答
- **保存内容**：LLM 完整对话历史（OpenAI 格式）+ UI 消息 + handler 状态（history_info + working）
- **跨进程恢复**：前端拿 sessionId → `store.load()` → 重建 client.history 和 handler

### 4.1 并发与 workspace 边界

- 单用户免登录模式默认保留可信 cwd 能力；生产可开启 `ENFORCE_WORKSPACE_FOR_SERVER`
- 开启 workspace 强制隔离后，`file_write` / `file_patch` / `code_run cwd` 必须在 `WORKSPACE_ROOT/<user_id>/<session_id>/` 内；越界返回 `workspace_violation`
- 长期记忆默认使用服务用户目录；开启用户 memory 配置后写入 `memory/users/<user_id>/`
- `MAX_GLOBAL_RUNS` / `MAX_USER_RUNS` 可限制同时运行的 agent run 数
- `RUNNER_BACKEND=thread` 保留进程内线程执行，适合本地开发
- `RUNNER_BACKEND=celery` 时 FastAPI 负责创建 run、读取 Redis Stream；agent run 由 `backend.celery_app` worker 执行
- Celery 模式使用 Redis 作为 broker、事件流、cancel 信号和 ask_user 回答通道，可支持多 Uvicorn worker；本阶段仍要求 SQLite 和 workspace 在同一台机器共享磁盘

### 5. 技能动态加载 `skill_manager.py`

**SOP-as-skill 模式**：`skills/<name>/SKILL.md` 文件本身既是文档又是可调用单元。

- 启动时 `scan_skills()` 扫描目录，解析每个 SKILL.md 的 frontmatter（name/description/usage）
- 注入一个虚拟工具 `use_skill`，schema 里枚举所有已发现的技能
- 用户输入 `/skill_name args...` 时直接命中（`match_skill`）；不带斜杠时由 LLM 自主调用 `use_skill`
- 命中后把 SKILL.md 全文塞到 next_prompt，agent 按手册执行

零代码加技能：写一个 markdown 文件即可。

---

## 两个入口

两个入口共享同一份 core，**核心模块完全不感知入口**——通过 `on_event` 回调和 `do_ask_user` 重写解耦。

### React `frontends/react`

Web 前端基于 Next.js App Router、Tailwind CSS 和 shadcn/ui。浏览器请求先到 Next route handlers，再由代理层转发到 `backend/server.py` 暴露的 Agent SSE、OpenAI compatible 接口和 session API。

- 免登录：浏览器请求不携带 bearer token，后端统一使用服务用户会话
- 会话列表：`GET/POST/DELETE /api/sessions` -> 代理到 `/v1/sessions`
- 对话：`POST /api/runs` 创建 run，`GET /api/runs/{run_id}/events` 消费结构化 SSE
- OpenAI 兼容：`POST /api/chat/completions`，保留给外部兼容客户端
- 暂停：`POST /api/sessions/{session_id}/cancel`
- 持久化：由后端统一写入 `SessionStore`，前端刷新后可恢复新格式消息和事件

### ACP `frontends/acp`

**[Agent Client Protocol](https://agentclientprotocol.com)** 适配器：JSON-RPC 2.0 over stdio。Zed / VSCode 等 IDE 启动 agent 进程后通过这条协议双向通信。

`frontends/acp/jsonrpc.py`（~80 行）实现 ndjson 双向 RPC：既响应入站 `session/prompt`，又能主动向 client 发 `fs/read_text_file` 让 IDE 代读文件。

**关键设计：**

- **挂起式 ask_user**：worker 线程跨 `session/prompt` RPC **存活**——`ask_user` 时 set `turn_done_evt`，本轮 RPC 返回 `stopReason=end_turn`；下次 `session/prompt` 到达时识别到 worker 还活着，把文本喂进 `ask_q` 唤醒，而不是新启 worker
- **文件操作委托**：声明 `clientCapabilities.fs` 的 client 会接管 file_read/write/patch；缺失能力时降级回本地 IO
- **结构化事件**：core 输出 `AgentEvent`，ACP 入口映射为 `session/update`；`sys.__stdout__` 只用于 JSON-RPC 报文，工具 `print()` 只进 stderr 调试日志
- **session 恢复**：声明 `loadSession` 能力，复用 `SessionStore` 还原历史

启动方式：`/frontends/acp/run.sh`（Zed 配置里指向它）。烟雾测试见 `tests/acp_smoke.py`。

---

## 记忆架构：四层自进化系统

mini-agent 最核心的设计。agent 不仅执行任务，还能从执行中学习并积累可复用的知识。

```
                  ┌──────────────────────────────────┐
                  │         System Prompt             │
                  │   sys_prompt.txt + L1 索引注入     │
                  └───────────────┬──────────────────┘
                                  │ agent 看到关键词
                                  ▼
┌─────────┐  导航  ┌──────────┐  按需读  ┌──────────┐
│ L1 索引  │──────▶│ L2 事实库 │──────────▶│ L3 SOP  │
│ ≤30 行  │       │ 环境事实  │          │ 任务手册 │
└─────────┘       └──────────┘          └──────────┘
     ▲                  ▲                     ▲
     │                  │                     │
     └──── background review ────────────────┘
              (异步调用 start_long_term_update)

                  ┌──────────────────────────────────┐
                  │  L4 原始会话归档（前端自动写入）     │
                  │  memory/L4_raw_sessions/{ts}.md   │
                  └──────────────────────────────────┘
```

**自进化闭环：**

1. **执行中** → agent 用 `update_working_checkpoint` 保存关键发现到短期工作记忆
2. **任务完成** → 主 run 先交付最终答案、完成状态落库、释放前端流
3. **后台 review** → 独立 review agent 基于完整会话快照判断是否有可沉淀经验，必要时调用 `start_long_term_update`
4. **结算流程** → `memory_management_sop.md` 决策树被注入为 next_prompt，review agent 在后台后续轮次中：
   - `file_read` 现有 L1/L2 → 用 `file_patch` 最小化更新 → 新场景则 `file_write` 新 L3 SOP
5. **下次任务** → L1 索引自动注入 system prompt → agent 发现关键词匹配 → 读取 L3 SOP → 按手册执行

**触发条件**（定义在 `backend/background_review.py` + `prompts/memory_management_sop.md`）：

- 发现了新的环境事实（路径、配置、用户偏好）
- 摸索出关键避坑点或非平凡步骤序列
- 过程值得跨会话复用，且不属于当前任务进度、临时 TODO、一次性结论或敏感信息

**四条核心公理**（定义在 `prompts/memory_management_sop.md`）：

1. **行动验证原则**：No Execution, No Memory —— 未经工具调用验证的信息禁止写入
2. **神圣不可删改性**：可以压缩/迁移，但绝不丢弃已验证的信息
3. **禁止存储易变状态**：不存时间戳、PID、临时路径
4. **最小充分指针**：上层只留能定位下层的最短标识

---

## 关键设计决策

### 1. next_prompt 作为行为控制面

工具返回的 `next_prompt` 不只是"提示"，它是控制 agent 后续行为的核心机制：

- `_anchor_prompt()` 注入工作记忆 → agent 不会忘记之前的发现
- `start_long_term_update` 注入整个 SOP 决策树 → 后台 review agent 按流程结算记忆
- `turn_end_callback` 注入 `[DANGER]` 警告 → 防止 agent 无效重试
- `do_ask_user` 注入用户回答 → 引导 agent 基于回答继续

这种"工具通过 prompt 引导 LLM"的模式，比在 system prompt 中写死所有规则更灵活——不同工具在不同时机注入不同的上下文。

### 2. 约定式工具分发 vs 注册表

选择 `getattr(self, f'do_{tool_name}')` 而非注册表/装饰器，原因：
- 代码量最小，加新工具零配置开销
- IDE 可直接跳转到 `do_xxx` 方法
- 工具 schema（`tools_schema.json`）和实现（`do_xxx`）保持 1:1 对应

### 3. 用 OpenAI SDK 而不是手写 HTTP

早期版本走 `requests + SSE` 是为了兼容多家 Anthropic 风格中转的特殊鉴权头。切到 OpenAI 兼容协议后这些复杂度全部消失：
- SDK 自动处理流式、重试、超时
- 消息格式标准化（user / assistant / tool 三种 role）
- Prompt cache 由服务端自动管，客户端零配置
- 换后端只改 `API_BASE` 和 `MODEL`

### 4. file_patch 的唯一性约束

`file_patch` 要求 `old_content` 在文件中**恰好出现 1 次**，否则拒绝修改。这是故意的——迫使 LLM：
- 修改前必须 `file_read` 获取最新内容
- 提供足够长的上下文确保匹配唯一性

代价是 LLM 偶尔需要多一轮交互，但避免了"改错位置"的严重后果。

### 5. 工作记忆的 skip 机制

`_anchor_prompt(skip=True)` 在同一轮的非首个工具调用时返回 `'\n'`。原因：
- OpenAI API 要求每个 assistant.tool_calls[i] 必须有对应的 tool message
- 但工作记忆只需注入一次，重复注入会浪费 token 且干扰 LLM

### 6. 历史摘要压缩

每轮从 LLM 回复中提取 `<summary>` 标签（一句话），存入 `history_info[]`。这比保留完整历史更高效：
- 工作记忆窗口只保留最近 20 条摘要
- 完整历史走 `client.history` + `trim_history` 做 token 级裁剪
- 两套压缩机制互补：摘要提供语义连续性，raw history 提供完整上下文

### 7. 跨任务状态传递的"提醒注入"

继承上一任务的 `key_info` 时，自动追加 `[SYSTEM] 此为 N 个对话前设置的 key_info`。这解决了一个微妙问题——agent 可能把旧任务的工作记忆当成当前任务的上下文，导致混淆。提醒注入让 agent 知道这些信息可能已过时，应该主动更新。

### 8. ACP 挂起式 ask_user

ACP 协议假设每个 `session/prompt` 都对应一次完整 turn。但 `ask_user` 需要打断 agent 等用户输入——这与 ACP 的请求-响应模型冲突。

解决：worker 线程**跨 RPC 存活**。`ask_user` 触发时 set `turn_done_evt` 让当前 RPC 返回；下次 `session/prompt` 到达时检测 worker 还在跑 → 视为对 `ask_user` 的回答，喂进 `ask_q` 而不是新启 worker。这让 ask_user 的暂停语义在 ACP 协议上自然落地，编辑器侧无需任何特殊适配。

---

## 扩展指南

### 加新工具

1. 在 `tools_schema.json` 加 JSON Schema 定义
2. 在 `GenericHandler` 加 `do_xxx(self, args, response) -> StepOutcome`
3. 没了

### 加新技能（SOP）

在 `skills/<name>/SKILL.md` 写一个 markdown：

```markdown
---
name: deploy_eas
description: 部署 EAS 服务的标准流程
usage: /deploy_eas <service_name>
---

# Steps
1. ...
```

下次启动自动加载，可通过 `/deploy_eas` 命令或 LLM 自主调用 `use_skill` 触发。

### 换 LLM 后端

改根目录 `.env` 的 `API_BASE` + `MODEL` + `API_KEY` 即可。常用端点和启动命令见 [README.md](./README.md)。

### 接其他 LLM SDK

实现一个类，提供 `chat(system, new_messages, tools) -> generator` 方法（`yield` 文本 chunk，`return Response(content, tool_calls, stop_reason)`），替换 `LLMClient` 即可。`new_messages` 走 OpenAI 标准格式。

### 加新前端

参考 `backend/agent_service.py` 或 `frontends/acp/server.py`：

- 继承 `GenericHandler` 覆盖 `do_ask_user`（queue 阻塞 / 网络 RPC / etc.）
- 用 `on_event` 回调消费结构化 AgentEvent
- 跨任务/多会话可复用 `SessionStore` 和独立 handler 上下文的组合模式

### 加可观测性

`agent_loop.py` 的 turn 边界天然就是 hook 点——每轮的 `tool_calls`、`tool_results`、`exit_reason` 都可收集；接 OpenTelemetry / Langfuse 都不需要改 core。

---

## 使用和启动

安装、配置、启动、API 调试和提交前检查集中维护在 [README.md](./README.md)。本文档只保留架构、设计决策和扩展开发说明。
