# MiniAgent 服务 API 文档

本文档面向服务部署后的调用方，说明如何通过 HTTP API 调用 Agent。示例中的 `BASE_URL` 请替换为实际部署地址。

```bash
# 本地默认（scripts/start.sh 启动）：后端 8683
export BASE_URL="http://127.0.0.1:8683"
```

如果部署平台在网关层要求鉴权，请按平台要求附加 `Authorization` 等请求头。当前后端服务本身不校验鉴权头。

> **重要**：`/v1/responses` 与 `/v1/chat/completions` **只支持流式**（`stream=true`）。传 `stream=false` 会得到 `400 unsupported_mode`。下文示例统一以 SSE 形式给出。
>
> **HITL 默认关闭**：新请求默认 `allow_hitl=false`，Agent 会自主继续执行；只有显式传 `allow_hitl=true` 时，`ask_user` 或需要人工审批的工具才会把本轮流终结为 `requires_action`。已经暂停的 HITL resume 请求会默认保持可中断。

## 目录

1. [Chat Completions](#1-chat-completions)
2. [Responses API](#2-responses-api)
3. [Sessions API](#3-sessions-api)
4. [其他接口](#4-其他接口)
5. [部署后 Smoke Test](#5-部署后-smoke-test)
6. [常见问题](#6-常见问题)

## 通用约定

- 请求体一律 JSON：`Content-Type: application/json`。
- 流式接口使用 SSE：`Content-Type: text/event-stream`，行末 `data: [DONE]` 结束流。
- 流式接口在长时间无新事件（默认 15 秒）时会发出 SSE 注释行 `: keepalive`（无 `event:` / `data:`，按 SSE 规范是注释），用于穿透中间代理的空闲连接探测。客户端**必须**忽略以 `:` 开头的行；几乎所有标准 SSE 解析器（含 OpenAI Python SDK、`fetch` + `EventSource`）会自动丢弃。
- 响应头中会带上：
  - `X-Session-Id`：当前会话 ID
  - `X-Run-Id`：当前运行 ID（仅在创建或订阅 Run 的接口里返回）
  - `Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no`（流式接口）
- `cwd` 字段可选，用于指定任务执行目录。开启 `ENFORCE_WORKSPACE_FOR_SERVER` 后，`cwd` 不能逃逸出每个 session 的 workspace。
- 所有 LLM 调用会按真实 token 数返回 `usage`（多轮内部调用会累加）。后端在 Agent 构造时强制 `model_settings.include_usage=true`，因此即便上游 provider（如 dashscope/qwen-plus）默认不返回 usage，本服务也会显式开启使其回流；只有当上游 LLM 完全不支持 `stream_options.include_usage` 时才会缺省。

### 错误格式

错误分两层：**HTTP 层**（接口本身被拒）走 `error` 字段；**Run 终止层**（接口接受了请求但 Agent 跑错了）由 Responses 终止事件 (`response.completed` 或 `response.failed`) 与 `GET /v1/responses/{id}` 的 `status`/`error` 字段表达。

#### HTTP 错误（请求层）

所有 4xx / 5xx 响应统一形如：

```json
{
  "error": {
    "message": "Run not found: run_xxx",
    "type": "invalid_request_error",
    "code": "run_not_found"
  }
}
```

包括校验失败（422）和未匹配的 404，都会被全局 exception handler 包装成同一 schema。`code` 取值枚举：

| HTTP | `error.code` | 含义 | 额外字段 |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | 请求体非法 / 缺字段（含 `model` 字段不合法） | — |
| 400 | `workspace_violation` | `cwd` 逃出允许范围 | — |
| 400 | `unsupported_mode` | `/v1/responses` 与 `/v1/chat/completions` 仅支持 `stream=true` | — |
| 400 | `invalid_resume` | 对暂停中的 response 未提交有效的 HITL resume 输入 | — |
| 404 | `response_not_found` | response id 不存在或不属于当前用户 | — |
| 404 | `""`（空 code） | 通用 not found（典型为 `session not found`） | — |
| 409 | `session_busy` | 该 session 上一轮还在运行 | `status`（session 当前状态） |
| 409 | `not_resumable` | `function_call_output` 指向的 response 当前不在 `requires_action` | — |
| 409 | `no_regeneratable_answer` | 该 session 没有可重生成的答案 | — |
| 413 | `request_too_large` | 请求体超过 `MAX_REQUEST_BODY_BYTES` | — |
| 422 | `validation_error` | FastAPI 路径 / 查询参数校验失败 | — |
| 429 | `capacity_exceeded` | 全局或单用户并发 run 超额 | `scope`、`limit` |
| 500 | `""` | 其他未捕获异常 | — |

#### Run 终态（执行层）

Run 由 OpenAI Agents SDK runner 驱动，每个 run 在 `agent_run_states` 表里有一行带 `status`、`run_state_blob`、`expires_at` 的记录。`/v1/responses`（以及共用同一 SSE 协议的 `/v1/chat/completions`）的终止事件就是该 run 的终态：

| 终态 status | 触发条件 | SSE 终止事件 | `error` 是否携带 |
| --- | --- | --- | --- |
| `completed` | Agent 正常返回最终消息 | `response.completed` | 否 |
| `requires_action` | 请求显式 `allow_hitl=true`，且 Agent 触发 `ask_user` 或带 `needs_approval` 的工具 | `response.requires_action` 中 `status: "requires_action"` + `required_action.submit_tool_outputs` | 否 |
| `failed` | 上游 LLM / 工具异常、`max_turns` 超限、workspace 越界等 | `response.failed` | 是（`error.message`） |
| `cancelled` | `POST /v1/sessions/{sid}/cancel` 主动取消 | `response.failed` 或客户端连接断开 | 否 |
| `expired` | `requires_action` 状态超过 7 天未恢复 | 后续 resume 请求返回 `409 not_resumable`；若 GC 已删除记录则返回 `404 response_not_found` | — |

**用法**：
- 只关心成功/失败 → 终止事件是 `response.completed` 即成功；`response.failed` 即失败。
- 普通多轮 → `POST /v1/responses` 带上一轮 `id` 作为 `previous_response_id`，`input` 继续传普通文本；服务端会基于上一轮 response 的历史启动一个新的 response。
- HITL 暂停 → 先在启动请求里传 `allow_hitl:true`；终止事件 status 为 `requires_action` 时，从 `required_action.submit_tool_outputs.tool_calls[]` 读 `call_id` + `function.arguments`，然后 `POST /v1/responses` 带 `previous_response_id` + `input=[{type:"function_call_output", call_id, output:"<answer>"}]` 续传同一个暂停 run。
- 想知道为什么失败 → `GET /v1/responses/{id}` 看 `status` + `error` 字段，或同一 run 的 audit 日志。

### 状态机

#### Session 状态

```
                    ┌───────── POST /v1/responses（新一轮）┐
                    │                                       │
                    ▼                                       │
          ┌────────────────┐                                │
   ──────▶│  idle / 终态  │                                │
   create └────────────────┘                                │
                  ▲                                         │
                  │                    ┌──────────────────┐ │
                  │                    │     running      │─┘
                  │   run 收尾         └──────────────────┘
                  └─────────────────────────────┘
                       (completed / failed /
                        cancelled — 三者都可
                        作为新一轮的起点)
```

可观察值：`idle` / `running` / `completed` / `failed` / `cancelled`。

合法转移：
- `idle` → `running`：`POST /v1/responses` 或 `/v1/chat/completions` 启动新 run
- `running` → `completed` / `failed` / `cancelled`：run 结束（详见上表）
- `completed` / `failed` / `cancelled` → `running`：同一 session 启动新一轮 run（终态都可作为新一轮起点）

> HITL 暂停不再表现为 session 状态。仅当请求显式 `allow_hitl=true`，且 Agent 触发 `ask_user`（或带 `needs_approval=True` 的工具）时，**Run** 才进入 `requires_action`，但 session 仍保持 `idle` —— 后续 resume 通过 `previous_response_id` + `function_call_output`（Responses 线）或 `role:"tool"` 消息（Chat 线）发起，无须独占 session。

#### Run 状态（SDK runner 生命周期）

```
   ┌──────────┐       ┌─────────────────────┐
   │ running  │──────▶│  requires_action    │──┐
   └──────────┘       └─────────────────────┘  │
        │                       │              │ resume：
        │                       │ 7d 未续答       │ previous_response_id +
        │                       ▼              │ function_call_output
        │                    expired           │  (or role:"tool")
        │                       │              ▼
        ▼                       │         ┌──────────┐
  ┌────────────────────────┐    │         │ running  │
  │ completed │ failed │   │    │         └──────────┘
  │ cancelled │ expired  │◀──┘                │
  └────────────────────────┘                  │
            ▲────────────────────────────────┘
```

可观察值：`running` / `requires_action` / `completed` / `failed` / `cancelled` / `expired`。

- `agent_run_states.status` 持久化上述值；`/v1/responses/{id}` 从这里取。
- `expired` 由后台 GC 在超过 `RUN_STATE_TTL_SECONDS`（默认 7 天）时自动写入；在此之后 resume 会得到 `409 not_resumable`，若记录已被硬删除则得到 `404 response_not_found`。
- `cancelled` 由 `POST /v1/sessions/{sid}/cancel`（取消其上最新 run）或客户端断流触发。

### 执行记录与后台 review

- `store=true`（Responses 默认）时，response 会写入 SQLite，供 `GET /v1/responses/{id}`、`previous_response_id` 多轮和 HITL resume 使用。
- 每轮会同步写入 session 的 `messages`，并把可回溯的 Markdown 副本保存到 `memory/L4_raw_sessions/`。
- 只有 `completed` 终态会在归档成功后触发后台 memory review；`requires_action`、`failed`、`cancelled` 不触发。
- 默认自主处理 HITL 的路径会记录 `hitl_auto_continue` 审计事件，便于回溯模型为什么没有中断等待用户。

## 推荐调用方式

按使用场景选择一套接口：

| 场景 | 推荐接口 |
| --- | --- |
| 只需要类似 OpenAI Chat Completions 的对话返回 | `/v1/chat/completions` |
| 需要结构化输出（工具调用、工具结果、显式开启的 HITL 暂停） | `/v1/responses` |

### 两个端点的关系

两个端点背后跑的是同一个 SDK Agent，区别只在 wire 形态：

| 维度 | `/v1/chat/completions` | `/v1/responses` |
| --- | --- | --- |
| 协议定位 | OpenAI Chat Completions 兼容 | OpenAI Responses 兼容 |
| 流式 chunk 形态 | 默认 `chat.completion.chunk.delta.content`；`allow_hitl=true` 暂停时可能出现 `tool_calls` | `response.output_text.delta` / `response.output_item.added` / `response.completed` |
| HITL 暂停表达 | `allow_hitl=true` 时：`finish_reason="tool_calls"` + 保留名 `__ask_user__` | `allow_hitl=true` 时：`response.requires_action` 内 `status="requires_action"` + `required_action.submit_tool_outputs` |
| HITL resume 方式 | 追加 `role:"tool"` 消息，同 `X-Session-Id` 重发 messages | `previous_response_id` + `function_call_output` 输入项 |
| 历史制品 | 仅靠 `X-Session-Id` 维持上下文 | `store=true` 默认开，`GET /v1/responses/{id}` 取回 / `DELETE` 删除 |
| 跨设备恢复 | 需要客户端自己回放最近 assistant 消息（含 `__ask_user__` tool_call） | `previous_response_id` 一手指针，跨设备/跨进程 |

怎么选：

- **`/v1/responses`**：首选。OpenAI Python SDK 直接 `responses.create(...)` 即可用；结构化事件完整；跨端恢复靠 `previous_response_id`。
- **`/v1/chat/completions`**：兼容只懂 Chat 协议的客户端。显式开启 `allow_hitl=true` 后，HITL 用保留名 `__ask_user__` 伪装成 tool_call，下一轮以 `role:"tool"` 续答。

> 之前曾对外暴露的 `/v1/runs` 一族（`POST /v1/runs`、`GET /v1/runs/{id}`、`/events`、`/stop`）已在 SDK 迁移收敛过程中整体删除；现存路径全部返回 404。前端改为直接订阅 `/v1/responses` 的 SSE。

## 1. Chat Completions

接口：

```http
POST /v1/chat/completions
```

> ⚠️ 仅支持流式（`stream=true`）。`stream=false` 会返回 `400 unsupported_mode`。

### 流式请求

```bash
curl --no-buffer --location "$BASE_URL/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --header 'X-Session-Id: demo-session-001' \
  --data '{
    "messages": [
      {"role": "user", "content": "检查当前目录有哪些文件，并简单总结"}
    ],
    "stream": true
  }'
```

流式返回遵循 OpenAI Chat Completions SSE 的文本增量格式（实测自 8683 后端）：

```text
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1779098260,"model":"qwen-plus","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1779098261,"model":"qwen-plus","choices":[{"index":0,"delta":{"content":"我是"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1779098261,"model":"qwen-plus","choices":[{"index":0,"delta":{"content":"通用执行 Agent"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1779098263,"model":"qwen-plus","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

说明：

- `/v1/chat/completions` 是面向通用 OpenAI 客户端的纯文本兼容入口：流式只输出 `delta.content`，不输出 `delta.tool_calls`，也不夹带任何自定义 SSE event。
- 末尾 chunk 的 `delta` 为空、`finish_reason="stop"`；Chat Completions 协议下 `usage` 不内嵌在末尾 chunk 里（不论上游是否返回）。如需 token 数请改用 Responses API 流末 `response.completed.usage`，那条**总会**带聚合后的真实 token（后端服务端强制开启 `include_usage`）。
- 需要在客户端看到工具调用、工具结果或 HITL `requires_action`，请改用 `/v1/responses`（结构化 SSE）。Chat 线只有在显式 `allow_hitl=true` 且实际暂停时，才会用 `__ask_user__` tool call 表达 HITL。
- `model` 字段：可省略；省略时使用 `/v1/models/active` 的当前值（本地默认 `qwen-plus`）。传入则**仅本次请求**透传给上游，不写入全局生效模型。具体路由到哪个上游 provider 由 `memory/runtime.json` 的前缀规则决定（详见下文"多 Provider / Key 池"）。
- `allow_hitl` 字段：可省略，默认 `false`。普通请求保持自主执行；只有显式 `true` 才允许流末 `finish_reason="tool_calls"` 暂停。以 `role:"tool"` 消息续答已暂停 run 时，服务端默认按 `allow_hitl=true` 处理。

## 2. Responses API

接口：

```http
POST /v1/responses
GET /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
```

适合需要结构化结果的调用方。流末 `response.completed`（或显式 `allow_hitl=true` 后可能出现的 `response.requires_action` / 出错时的 `response.failed`）的 payload 里会包含最终消息、工具调用、工具输出。

> ⚠️ 仅支持流式（`stream=true`）。`stream=false` 会返回 `400 unsupported_mode`。`output` 在流式过程中通过 `response.output_item.added/done` 增量出现，最终 payload 只在终止事件中给出完整列表。

### 流式请求

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "input": "检查当前目录有哪些文件，并简单总结",
    "stream": true
  }'
```

流式事件（实测自 8683 后端，每个 event 都带 `sequence_number` 单调递增）：

| 事件 | 含义 |
| --- | --- |
| `response.created` | Response 已创建（顶层 `id`、`status:"in_progress"`、`model`） |
| `response.reasoning_step.started` | 合成的"思考"段开始；`step_id` 形如 `rs_synth_<resp_id>_<n>`，带 `synthetic:true`。**仅本服务端合成**，OpenAI 原生 Responses 协议没有此事件 |
| `response.output_item.added` | 新增输出项 —— 可能是 assistant `message`，也可能是 `function_call` |
| `response.output_text.delta` | 最终消息文本增量（`delta` 是当次新增 token） |
| `response.function_call_arguments.delta` | 工具调用 `arguments` 字段增量（按 `item_id` 关联到 `output_item.added` 的 `function_call`） |
| `response.function_call_arguments.done` | 工具调用 `arguments` 累计完成 |
| `response.output_text.done` | 最终消息文本完成（顶层 `text` 是完整内容） |
| `response.output_item.done` | 某个输出项结束（`item.status:"completed"`） |
| `response.reasoning_step.completed` | "思考"段结束 |
| `response.requires_action` | **PAI-RAG 扩展事件**：仅显式 `allow_hitl=true` 后可能出现的 HITL 暂停（payload `status:"requires_action"` + `required_action.submit_tool_outputs`，仿 Assistants v1）。`GET /v1/responses/{id}` 也会以 `status:"requires_action"` 返回相同 payload，可作为跨设备兜底 |
| `response.incomplete` | **OpenAI 标准终止事件**：紧跟在 `response.requires_action` 之后再发一次（payload 同上，但 `status="incomplete"` + `incomplete_details:{reason:"requires_action"}`）。OpenAI 官方 SDK 默认只监听 `response.completed/failed/incomplete` 三个终态 event，所以**纯 OpenAI SDK 客户端可以靠 `response.incomplete` 关闭连接**，再用 `GET /v1/responses/{id}` 或上一条 `response.requires_action` 拿到 `required_action` 字段。重度依赖 HITL 的客户端建议直接 hook `response.requires_action` |
| `response.completed` | Response 正常完成（流末，包含完整 `output` 与 `usage`） |
| `response.failed` | Response 失败（流末，含 `error.message`） |

示例（最简文本回答，简化版）：

```text
event: response.created
data: {"type":"response.created","id":"resp_xxx","status":"in_progress","model":"qwen-plus","output":[],"usage":null,"sequence_number":1}

event: response.reasoning_step.started
data: {"type":"response.reasoning_step.started","response_id":"resp_xxx","step_id":"rs_synth_resp_xxx_1","synthetic":true,"sequence_number":2}

event: response.output_item.added
data: {"type":"response.output_item.added","response_id":"resp_xxx","output_index":0,"item":{"id":"msg_xxx","type":"message","status":"in_progress","role":"assistant","content":[]},"sequence_number":3}

event: response.output_text.delta
data: {"type":"response.output_text.delta","response_id":"resp_xxx","delta":"我是","output_index":0,"content_index":0,"sequence_number":5}

event: response.output_text.done
data: {"type":"response.output_text.done","response_id":"resp_xxx","text":"我是通用执行 Agent...","output_index":0,"content_index":0,"sequence_number":20}

event: response.output_item.done
data: {"type":"response.output_item.done","response_id":"resp_xxx","output_index":0,"item":{"id":"msg_xxx","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"我是通用执行 Agent..."}]},"sequence_number":21}

event: response.reasoning_step.completed
data: {"type":"response.reasoning_step.completed","response_id":"resp_xxx","step_id":"rs_synth_resp_xxx_1","synthetic":true,"sequence_number":22}

event: response.completed
data: {"type":"response.completed","id":"resp_xxx","status":"completed","model":"qwen-plus","output":[{"id":"msg_xxx","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"我是通用执行 Agent..."}]}],"usage":{"input_tokens":3150,"output_tokens":10,"total_tokens":3160,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}},"sequence_number":23}

data: [DONE]
```

工具调用流（实测：显式 `allow_hitl:true` 后触发 `ask_user`），关键节选：

```text
event: response.output_item.added
data: {"type":"response.output_item.added","response_id":"resp_xxx","output_index":1,"item":{"arguments":"","call_id":"call_xxx","name":"ask_user","type":"function_call","id":"call_xxx"},"sequence_number":6}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","response_id":"resp_xxx","item_id":"call_xxx","output_index":1,"delta":"{\"question\": \"","sequence_number":9}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","response_id":"resp_xxx","item_id":"call_xxx","output_index":1,"delta":"请提供要操作的文件名？","sequence_number":11}

event: response.output_item.done
data: {"type":"response.output_item.done","response_id":"resp_xxx","output_index":1,"item":{"arguments":"{\"question\": \"请提供要操作的文件名？\"}","call_id":"call_xxx","name":"ask_user","type":"function_call","id":"call_xxx"},"sequence_number":16}

event: response.requires_action
data: {"type":"response.requires_action","id":"resp_xxx","status":"requires_action","required_action":{"type":"submit_tool_outputs","submit_tool_outputs":{"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"ask_user","arguments":"{\"question\": \"请提供要操作的文件名？\"}"}}]}},"sequence_number":23}

event: response.incomplete
data: {"type":"response.incomplete","id":"resp_xxx","status":"incomplete","incomplete_details":{"reason":"requires_action"},"required_action":{"type":"submit_tool_outputs","submit_tool_outputs":{"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"ask_user","arguments":"{\"question\": \"请提供要操作的文件名？\"}"}}]}},"sequence_number":24}

data: [DONE]
```

> 兼容性提示：上游 provider（如 qwen-plus）会对 `function_call` 项发回固定占位 id `__fake_id__`。事件桥会把它**就地改写为以 `call_id` 派生的 `id`** 后再下发，因此客户端可以放心按 `item_id` / `item.id` 去重和聚合 arguments delta；终止事件 `response.completed.output` / `response.requires_action.output` 中的 `function_call` 条目都已带稳定 `id` 和 `call_id`，不会再出现 `__fake_id__`。

> 工具结果消毒：服务端在把工具的 raw stdout / 内部状态打回到 `function_call_output` 项之前，会自动剥离内部脚手架（如 `### [WORKING MEMORY]` / `<history>` 块）。客户端拿到的 `output` 字段都是给用户看的最终内容，不需要再做二次清洗。
>
> 函数参数 JSON 校验：流式过程中 `response.function_call_arguments.delta` 拼接出来的 `arguments` 应当是合法 JSON。事件桥会在 `response.function_call_arguments.done` / 对应 `response.output_item.done` 时校验一次；若解析失败（典型场景：上游模型 truncate 或乱码），该 function_call item 会被标记成 `status: "incomplete"` + `arguments_status: "invalid_json"`，对应 `done` 事件载荷也会带上同样字段。客户端可据此跳过执行 / 触发兜底。`status="completed"` 且没有 `arguments_status` 字段即视为 valid。

### 多轮和会话关联

> **业务集成建议**：对外只用两种方式串多轮：
> - `previous_response_id`（OpenAI 标准；指向上一轮 response，复用其 session/历史/`conversation`/`instructions`）
> - `conversation`（业务方自己的会话串号；服务端按其下"最新一条 response"作隐式 previous）
>
> 下方 §session_id / §conversation_history / `X-Session-Id` 头属于 PAI-RAG 内部 / 历史接口，仅为前端 / 老客户端保留，**不建议外部业务方继续依赖**——这两条路径未来可能合并到 `conversation` 一条线下。

外部业务方推荐字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `input` | string / array | 当前输入，必填 |
| `model` | string | 模型 id：仅本次请求覆盖上游 LLM；不写入全局生效模型，下一次不传则回到服务端当前生效模型（`/v1/models/active` 的值，回退顺序：运行时值 > 环境变量 `MODEL` > `qwen-plus`）。具体路由到哪个 provider 由 `memory/runtime.json` 的前缀规则决定（详见下文"多 Provider / Key 池"） |
| `instructions` | string | 本轮系统级说明 |
| `previous_response_id` | string | OpenAI 标准多轮指针：继续某个历史 response |
| `conversation` | string | 业务方自定义会话标识；服务端按其下最新 response 作隐式 previous |
| `stream` | boolean | 是否流式返回；**当前仅支持 `true`**，传 `false` 返回 `400 unsupported_mode` |
| `allow_hitl` | boolean | 是否允许 Agent 暂停等待用户输入。默认 `false`；显式 `true` 后才会产生 `requires_action`。HITL resume 输入默认按 `true` 处理 |
| `store` | boolean | 是否保存 response（用于后续 `GET` / `previous_response_id`），默认 `true` |
| `cwd` | string | 任务执行目录 |

**Legacy / 内部字段（不推荐外部业务方使用，保留是为了前端 / 老客户端兼容）**：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `session_id` | string | 服务端 session 行的引用（前端会话管理用）；外部业务方请改用 `conversation` 串号 |
| `conversation_history` | array | 无状态模式下显式传入的历史消息；外部业务方请改用 `previous_response_id` 让服务端自己取上下文 |
| `X-Session-Id`（HTTP 头） | string | 仅 `/v1/chat/completions` 路径上保留；Responses 路径不读取此头 |

#### 方式一：使用 `previous_response_id`

适合调用方不想额外维护服务端 session，只想基于上一轮 response 继续对话的场景。

第一轮：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，你是谁？","stream":true}'
```

从流末 `response.created` / `response.completed` 事件里记录 `id`（也可监听 `response.created` 事件第一时间拿到）：

```json
{"id": "resp_xxx", "object": "response", "status": "completed"}
```

第二轮带上 `previous_response_id`：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"previous_response_id":"resp_xxx","input":"那你能帮我做什么？","stream":true}'
```

服务会复用上一轮 response 绑定的会话和历史上下文。

#### 方式二：使用 `session_id`（Legacy / 内部）

> **不推荐外部业务方使用**。该字段对应服务端 `sessions` 表的一行，主要服务于 PAI-RAG 自己的前端会话管理；外部业务方请改用方式三 `conversation` 串号。下面示例仅为兼容老客户端保留。

先创建 session：

```bash
SESSION_ID=$(curl -s -X POST "$BASE_URL/v1/sessions" \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
```

第一轮：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"你好，你是谁？\",\"stream\":true}"
```

第二轮继续传同一个 `session_id`：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"继续刚才的话题\",\"stream\":true}"
```

#### 方式三：使用 `conversation`

适合调用方已经有自己的业务会话 ID，希望用业务 ID 串起多轮 response 的场景。

第一轮：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"conversation":"user-123-ticket-456","input":"第一轮问题","stream":true}'
```

第二轮继续传同一个 `conversation`：

```bash
curl --no-buffer "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"conversation":"user-123-ticket-456","input":"第二轮问题","stream":true}'
```

推荐选择：

| 场景 | 推荐方式 |
| --- | --- |
| 后端集成，想最少维护状态 | `previous_response_id`（OpenAI 标准） |
| 调用方已有自己的业务会话 ID，想串多轮 | `conversation` |
| 前端 / 老客户端要明确创建、查询、删除服务端会话 | `session_id`（Legacy / 内部） |

#### 优先级与互斥关系

一次请求可以同时携带多个上下文字段，服务端按下面的顺序解析：

1. **`previous_response_id`**（OpenAI 标准）：若给出，默认是普通多轮 continuation；服务端会加载该 response 的 session、对话历史、`instructions`、`conversation` 作为本轮默认值，并为本轮生成新的 response id。只有当 `input` 是 `function_call_output` / `mcp_approval_response` 时，才按 HITL resume 处理，恢复同一个暂停 run。
2. **`conversation`**（业务串号，仅在没有 `previous_response_id` 时生效）：查找该 `conversation` 标识下最新一条 response 作为隐式 previous。
3. **`session_id`**（Legacy / 内部）：始终是显式优先；若没传，则继承自第 1/2 步推导出的 previous response。外部业务方不要主动传。
4. **`conversation_history`**（Legacy / 内部，无状态模式）：若传了，直接覆盖从 1/2 推出的历史。外部业务方请改用 `previous_response_id`，让服务端自己取上下文，避免每轮都把整段历史重传。
5. **`instructions`**：本轮显式 `instructions` 覆盖 previous response 上的 `instructions`。

读取 response：

```bash
curl --location "$BASE_URL/v1/responses/resp_xxx"
```

删除 response：

```bash
curl --request DELETE --location "$BASE_URL/v1/responses/resp_xxx"
```

### 按 response_id 取消 run

```bash
curl --request POST --location "$BASE_URL/v1/responses/resp_xxx/cancel"
```

成功返回：

```json
{"id": "resp_xxx", "object": "response", "status": "cancelled"}
```

行为：

- **正在运行**：服务端通过内存中的 in-flight 注册表把 `resp_xxx` 映射到所属 session，再转发到 `service.cancel_session(...)`；此时连接到该 response 的 SSE 流会立刻收到 `response.failed`（或客户端断流）。
- **已落库的终态 response**（completed/failed/cancelled）：通过 `agent_run_states` 取出 session id 再 cancel；如该 session 当前没有活动 run，本调用是 idempotent 的，仍返回 `status:"cancelled"`。
- **不存在 / 不属于本用户**：返回 `404 response_not_found`。

与 `POST /v1/sessions/{sid}/cancel` 的区别：cancel-by-session 适合调用方手里只有 session id 的情况；cancel-by-response 是 OpenAI Responses 标准入口，适合直接在 Responses API 流的 `response_id` 上做超时/中断。两者底层走同一个 `cancel_session` 路径，因此对同一活动 run 来说是等价的。

### HITL：暂停 → 询问用户 → 恢复

默认新请求不会暂停等待用户：`allow_hitl=false` 时，`ask_user` 会按工具参数里的 `default_action` 或安全兜底指令继续执行；其他需要审批的工具会被保守拒绝并要求模型给出可读报告。该自动处理路径会写入 `hitl_auto_continue` 审计事件。

需要产品形态上弹出用户确认/补充输入时，在启动请求里显式传 `allow_hitl:true`：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "input": "执行前如缺少关键信息，请暂停询问我",
    "allow_hitl": true,
    "stream": true
  }'
```

当 Agent 调用 `ask_user`（或任何带 `needs_approval=True` 的工具）时，本轮 response 在流末连发两条事件 —— 业务自有事件 `response.requires_action` + OpenAI 标准事件 `response.incomplete`，然后 `data: [DONE]`：

```text
event: response.requires_action
data: {"type":"response.requires_action",
       "id":"resp_pause",
       "status":"requires_action",
       "required_action":{
         "type":"submit_tool_outputs",
         "submit_tool_outputs":{
           "tool_calls":[{
             "id":"call_ask_001",
             "type":"function",
             "function":{
               "name":"ask_user",
               "arguments":"{\"question\":\"请选择 A 或 B？\"}"
             }
           }]
         }
       }}

event: response.incomplete
data: {"type":"response.incomplete",
       "id":"resp_pause",
       "status":"incomplete",
       "incomplete_details":{"reason":"requires_action"},
       "required_action":{...同上...}}

data: [DONE]
```

- 服务端把此时的 SDK `RunState.to_string()` blob 持久化到 `agent_run_states`，`status='requires_action'`，TTL 默认 7 天。
- `GET /v1/responses/resp_pause` 任何时间都能读到 `status:'requires_action'` + 上述 `required_action`，跨设备/跨进程都可见。
- 关于双终止事件：`response.requires_action` 是 PAI-RAG 扩展，业务客户端可以直接消费它拿到 `required_action`；`response.incomplete` 是 OpenAI Responses 标准终止事件，纯 OpenAI SDK 默认只监听 completed/failed/incomplete，所以这条让纯 SDK 客户端能正常关闭流。客户端只需挑一条消费即可，不要把它们当作两次独立终止。

恢复 —— 客户端再发一次 `POST /v1/responses`，带上 `previous_response_id` 与 `function_call_output`：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "previous_response_id": "resp_pause",
    "input": [{
      "type": "function_call_output",
      "call_id": "call_ask_001",
      "output": "A"
    }],
    "stream": true
  }'
```

服务端按 `previous_response_id` 取出 RunState、`approve()` 对应的 `ToolApprovalItem`，然后用同一个 SDK runner `Runner.run_streamed(state, ...)` 续跑；从客户端视角，这就是 OpenAI Responses API 的标准 `submit_tool_outputs` 续传。

错误码：

- 同一 `previous_response_id` 已被其它客户端续过，或该 response 已不在 `requires_action` → `409 not_resumable`。
- 暂停超过 `RUN_STATE_TTL_SECONDS`（默认 7 天）且 RunState 已被清理 → `404 response_not_found`。
- 对仍处于 `requires_action` 的 response 发送普通文本，而不是 `function_call_output` / `mcp_approval_response` → `400 invalid_resume`。
- `previous_response_id` 不属于当前 `user_id` → `404 response_not_found`。

`/v1/chat/completions` 的等价形态：启动请求同样需要 `allow_hitl:true`。暂停时流末返回 `finish_reason="tool_calls"` + 一个保留名 `__ask_user__` 的 tool_call（`tool_call_id` 与 Responses 线路一致）；恢复时追加 `{"role":"tool","tool_call_id":"call_ask_001","content":"A"}`，带同 `X-Session-Id` 重发 messages 即可。两条线路共享同一份 `agent_run_states` 行，因此可以一端暂停、另一端恢复。

## 3. Sessions API

Sessions 用于服务端多轮会话管理。Chat Completions 通过 `X-Session-Id` 复用会话；Responses 通过请求体里的 `session_id` 复用会话。

> ⚠️ 服务端目前不校验调用方对 session 的所有权——任何持有 `session_id` 的请求都可读/写/删除它。需要多租户隔离时，应在网关层加签名或鉴权头。

### 创建 Session

```bash
curl --request POST --location "$BASE_URL/v1/sessions" \
  --header 'Content-Type: application/json' \
  --data '{}'   # 可选传 {"cwd": "..."} 指定执行目录
```

返回：

```json
{
  "session_id": "session_xxx",
  "title": "New Task",
  "created_at": "2026-05-13T10:56:53.980556",
  "updated_at": "2026-05-13T10:56:53.980556",
  "status": "idle",
  "active_run_id": "",
  "messages": []
}
```

时间字段是 ISO-8601 字符串；`status` 取值 `idle` / `running` / `completed` / `failed` / `cancelled`，状态转移见 §通用约定 → §状态机。HITL 暂停不再表现在 session 状态上 —— 该层语义迁移到 run 状态 `requires_action`。

### 列表与查询

```bash
curl --location "$BASE_URL/v1/sessions"
```
返回 `{"object":"list","data":[...]}`，目前**不支持** `limit` / `after` / `order` 等分页参数，会一次性返回所有 session 的 metadata。

```bash
curl --location "$BASE_URL/v1/sessions/session_xxx"
```
返回字段与创建相同。`active_run_id` 不为空表示有 run 正在运行。

### 删除 Session

```bash
curl --request DELETE --location "$BASE_URL/v1/sessions/session_xxx"
```
返回 `{"deleted":true,"session_id":"session_xxx"}`。

### 取消当前 Session 的活动 Run

```bash
curl --request POST --location "$BASE_URL/v1/sessions/session_xxx/cancel"
```
返回 `{"cancelled":true,"session_id":"session_xxx"}`。

### 重新生成上一轮答案

```bash
curl --no-buffer --request POST --location "$BASE_URL/v1/sessions/session_xxx/regenerate" \
  --header 'Content-Type: application/json' \
  --data '{}'
```

返回 `text/event-stream` —— 即与 `POST /v1/responses` 完全一致的 SSE 流（最后一行 `data: [DONE]`）。服务端会先把 session 内最后一条 assistant 消息修剪掉，然后用上一条 user 消息重发一次 SDK run。

若该 session 没有可重生成的答案，返回 409 `no_regeneratable_answer`。

### 错误码补充

完整错误码枚举见 §通用约定 → §错误格式。Sessions API 上常见的几类：

| HTTP | `error.code` | 触发条件 |
| --- | --- | --- |
| 404 | — | session 不存在 |
| 409 | `session_busy` | 该 session 上一轮还在运行，无法启动新 run |
| 409 | `no_regeneratable_answer` | 该 session 没有可重生成的答案 |
| 429 | `capacity_exceeded` | 全局或单用户并发 run 超 `MAX_GLOBAL_RUNS` / `MAX_USER_RUNS` |

## 4. 其他接口

### 健康检查

```bash
curl --location "$BASE_URL/health"          # {"status":"ok"}
curl --location "$BASE_URL/health/detailed" # 见下方示例
```

`/health/detailed` 返回示例：

```json
{
  "status": "ok",
  "runner_backend": "sdk",
  "model": "qwen-plus",
  "checks": {
    "api": "ok",
    "sqlite": "ok",
    "redis": "disabled",
    "runner": "sdk"
  }
}
```

整体 `status` 为 `degraded` 表示有依赖检查失败（具体看 `checks` 各字段）。

### 模型列表

```bash
curl --location "$BASE_URL/v1/models"
```

返回 OpenAI 兼容结构，并额外携带 `active_model` 顶层字段以及 `data[].active` 标记：

```json
{
  "object": "list",
  "active_model": "qwen-plus",
  "data": [
    {"id": "qwen-plus", "object": "model", "created": 1778640000, "owned_by": "pai-rag", "active": true},
    {"id": "mini-agent", "object": "model", "created": 1778640000, "owned_by": "pai-rag", "active": false},
    {"id": "agent", "object": "model", "created": 1778640000, "owned_by": "pai-rag", "active": false}
  ]
}
```

- `active_model`：当前实际生效的模型，所有新建的对话默认走它。优先级：运行时切换值（见下文）> 环境变量 `MODEL` > `qwen-plus`。
- `data`：当前生效模型 + `MODEL_ALIASES`。第一项始终是 active；`data[].active=true` 标记当前生效项。
- `created` 是请求时刻的时间戳。

### 切换当前生效模型

`POST /v1/models/active` 修改全局生效模型。改动**立即对所有新建对话生效**（已经在跑的 stream 沿用原模型），并写入 `memory/runtime.json` 的 `active_model` 字段跨进程 / 跨重启持久化（与多 provider / key 池配置共用同一文件，写入时 read-modify-write 不会动 `providers` 段）。HTTP server / Celery worker / ACP server 共享同一份。

```bash
curl --location -X POST "$BASE_URL/v1/models/active" \
  --header 'Content-Type: application/json' \
  --data '{"model": "qwen-max"}'
```

成功返回：

```json
{"active_model": "qwen-max"}
```

请求体只接受 `{"model": "<name>"}` 一种 schema。`<name>` 校验规则：

- 必须是非空字符串
- 不能含空白字符（空格 / tab / 换行）
- 长度 ≤ 200

不满足返回 `400`：

```json
{"error": {"message": "model must not contain whitespace", "type": "invalid_request_error", "code": "invalid_request_error"}}
```

> 注意：服务端不会预先校验上游 LLM 是否真的支持该模型名。如果输错，下次对话调用上游 API 时才会失败（透传上游错误）。

回退到环境变量默认值：直接 `POST` 当前 `MODEL` 即可，没有专门的 reset 接口。

### 多 Provider / Key 池

服务端通过 `memory/runtime.json` 配置多个上游 provider 和每个 provider 下的 key 池，运行时按 model 名前缀路由、按 round-robin 分配 key、按返回状态码自动维护 key 健康度。无此文件、或文件中没有 `providers` 段时回退到环境变量 `API_KEY` / `API_BASE` 拼一个默认 `qwen` provider，行为与单 key 单 base 时完全一致——这是默认状态，不写文件即可。

#### 配置文件 schema

`memory/runtime.json`（同时承载 `/v1/models/active` 写入的 `active_model` 字段）：

```json
{
  "active_model": "qwen-plus",
  "default_provider": "qwen",
  "cooldown_seconds": 300,
  "providers": {
    "qwen": {
      "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_keys": ["sk-aaa", "sk-bbb"],
      "model_prefixes": ["qwen-", "qwq-"]
    },
    "deepseek": {
      "api_base": "https://api.deepseek.com",
      "api_keys": ["sk-ds-1"],
      "model_prefixes": ["deepseek-"]
    },
    "zhipu": {
      "api_base": "https://open.bigmodel.cn/api/paas/v4",
      "api_keys": ["zp-1"],
      "model_prefixes": ["glm-"]
    }
  }
}
```

字段说明：

- `active_model`：全局生效模型，由 `/v1/models/active` 维护；可手工预置但通常通过接口写入。
- `default_provider`：未匹配到任何 `model_prefixes` 时使用的 provider。
- `cooldown_seconds`：429 触发的 key 冷却时长，默认 300 秒。
- `providers.<name>.api_base`：该 provider 的 OpenAI 兼容 base URL。
- `providers.<name>.api_keys`：key 池数组，按顺序 round-robin。
- `providers.<name>.model_prefixes`：模型名前缀列表，用于路由。例 `qwen-max` → `qwen`，`glm-4-plus` → `zhipu`。

文件 `mtime` 变化即热加载，无需重启进程；HTTP server / Celery worker / ACP server 各自维护一份内存状态（不跨进程共享 key 健康度）。`/v1/models/active` 写入时采用 read-modify-write，仅更新 `active_model` 字段，不会动 `providers` 段。

#### Key 健康度与失败规则

每次上游调用失败，按 HTTP 状态码处理对应 key：

| 状态码 | 行为 | 何时复活 |
| --- | --- | --- |
| 401 / 403 | 永久 evict（认为 key 失效） | 重启进程 / 改 `runtime.json` 触发重载后 |
| 429 | 进入 `cooldown_seconds` 秒冷却 | 冷却到期时下次 `acquire` 自动复活，无需 probe |
| 5xx | 不影响 key 池（透传） | — |
| 其他 | 不影响 key 池（透传） | — |

LLMClient 在 401/403/429 时会自动从 key 池取下一把 key 重试一次（最多 2 次轮换），全部 evict 时回退到环境变量 `API_KEY` / `API_BASE`，仍失败则按原始错误透传给调用方。

#### 路由优先级

```
请求里的 body.model
  ↓ （前缀匹配）providers[*].model_prefixes
  ↓ 命中 → 该 provider 的 key 池
  ↓ 未命中 → default_provider 的 key 池
  ↓ key 池全部 evict → env API_KEY / API_BASE 兜底
```

#### 安全说明

- `runtime.json` 含明文 key，**严禁入库**（仓库 `.gitignore` 已覆盖 `memory/`）。
- 服务端日志和事件流不会输出完整 key，最多输出 key 末 4 位用于排查。
- 上线建议：先从单 provider 单 key 起步，验证 `/v1/models/active` 切换无误后再扩到多 key / 多 provider。

### Skills 列表

```bash
curl --location "$BASE_URL/v1/skills"
```
列出 `skills/` 目录下可被 `use_skill` 工具调用的技能。

## 5. 部署后 Smoke Test

仓库提供了一个纯 Python 标准库脚本（`scripts/api_smoke.py`），用来验证部署后的两类 API 是否可用：

- `/v1/chat/completions`（流式）
- `/v1/responses`（流式）

> 当前两个端点都只接受 `stream=true`，所以脚本内部统一走 SSE 消费；`--base-url` 默认指向本地 `http://127.0.0.1:8683`。

默认 sample query：

```text
你好，请用一句话介绍你自己，并说明你可以通过 API 被调用。
```

最简跑法（直接打本地 8683）：

```bash
python scripts/api_smoke.py
```

打到其它部署：

```bash
python scripts/api_smoke.py --base-url "$BASE_URL"
```

如果部署网关需要鉴权：

```bash
python scripts/api_smoke.py --base-url "$BASE_URL" --auth "Bearer xxx"
```

如果鉴权头不是 `Authorization`，可以使用自定义 header（可重复传）：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --header "Authorization: your-token" \
  --header "X-Request-Id: smoke-test-001"
```

只测试某一类 API：

```bash
python scripts/api_smoke.py --only chat
python scripts/api_smoke.py --only responses
```

自定义 sample query：

```bash
python scripts/api_smoke.py --query "你好，请说明你是什么服务"
```

显式指定模型（默认不传，由后端使用 `/v1/models/active` 的当前值）：

```bash
python scripts/api_smoke.py --model qwen-max
```

测试 Responses API 的 `previous_response_id` 多轮：

```bash
python scripts/api_smoke.py --only responses --multi-turn
```

脚本成功时会打印每类 API 的事件总数、`response_id` 以及解析出的最终答案文本。

业务配置校验类请求也可以用同一个脚本验证调用链是否通。示例：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --only responses \
  --timeout 600 \
  --query "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。请列出匹配配置版本，获取 Released 配置并运行配置校验；最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。"
```

注意：HTTP/API 层成功表示 Agent run 正常完成；业务配置是否校验通过要看最终答案。配置存在错误时，流末仍可能是 `response.completed`，最终文本会报告“校验失败”和错误摘要。

## 6. 常见问题

### 为什么 Chat Completions 流式没有 `delta.tool_calls`？

Agent 的工具调用是服务端自动完成的，不要求客户端回传工具结果。`/v1/chat/completions` 定位是给通用 OpenAI 客户端使用的纯文本兼容入口，所以默认不输出 `delta.tool_calls`，也不带任何自定义事件。需要看到工具调用、工具结果或 HITL `requires_action`，请改用 `/v1/responses`。唯一例外是显式 `allow_hitl=true` 且确实暂停时，Chat 流末会带保留名 `__ask_user__` 的 tool_call。

### 什么时候用 `session_id`？

需要多轮上下文时使用。调用方可以：

- Chat Completions：传 `X-Session-Id`
- Responses：传 `session_id` 或 `previous_response_id`

### 为什么传 `stream=false` 会被拒绝？

`/v1/responses` 与 `/v1/chat/completions` 当前**只支持流式**。早期文档曾给出非流式示例，已不再可用：服务端会返回 `400 unsupported_mode`。原因是后端把 SDK runner 输出的事件流（工具调用、HITL 暂停、token 增量）作为一等公民暴露，非流式聚合视图维护成本高且不符合实际使用形态；客户端只需消费 SSE 的最终 `response.completed` 事件即可拿到与历史"非流式"等价的完整 payload。

### SSE 中的 keepalive 是什么？

流式接口在 15 秒无新 event 时会发送一行 SSE 注释：

```text
: keepalive
```

按 SSE 规范，以 `:` 开头的行属于注释，标准 SSE 解析器（OpenAI Python SDK、浏览器 `EventSource`、`sseclient` 等）会自动丢弃。如果调用方是手写的逐行解析（split by `\n\n`），需要在每行前判断 `if line.startswith(':'): continue`。这条注释只为穿透中间代理的空闲超时（典型 60s），并不代表流出错或卡住，也不会重置事件序号。

### 怎么拿到真实 token 使用量？

- Responses（流式，推荐）：最后一条 `response.completed` 事件里的 `usage`。
- Chat Completions（流式）：当前**未在末尾 chunk 里内嵌** `usage`；如需 token 数请改用 Responses 流。

`usage` 是本次请求 Agent 所有内部 LLM 调用的累加值。后端会强制对所有 chat-completions 兼容上游开启 `stream_options.include_usage=true`（包括 dashscope/qwen-plus 这类 SDK 默认不开的 provider），因此 `response.completed.usage` 在正常完成时**总会**有真实 token。仅当任务失败、被取消，或上游 LLM 完全不支持 usage 时才可能为 `null`。

### 单一 session 能并发跑多个 run 吗？

不能。同一 `session_id` 同一时刻只允许一个 active run；上一轮还在跑时再发 `POST /v1/responses` 会返回 409 `session_busy`。需要并发请走不同的 session。
