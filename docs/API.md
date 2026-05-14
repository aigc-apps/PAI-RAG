# MiniAgent 服务 API 文档

本文档面向服务部署后的调用方，说明如何通过 HTTP API 调用 Agent。示例中的 `BASE_URL` 请替换为实际部署地址。

```bash
export BASE_URL="http://127.0.0.1:8000"
```

如果部署平台在网关层要求鉴权，请按平台要求附加 `Authorization` 等请求头。当前后端服务本身不校验鉴权头。

## 目录

1. [Chat Completions](#1-chat-completions)
2. [Responses API](#2-responses-api)
3. [Runs API](#3-runs-api)
4. [Sessions API](#4-sessions-api)
5. [其他接口](#5-其他接口)
6. [部署后 Smoke Test](#6-部署后-smoke-test)
7. [常见问题](#7-常见问题)

## 通用约定

- 请求体一律 JSON：`Content-Type: application/json`。
- 流式接口使用 SSE：`Content-Type: text/event-stream`，行末 `data: [DONE]` 结束流。
- 响应头中会带上：
  - `X-Session-Id`：当前会话 ID
  - `X-Run-Id`：当前运行 ID（仅在创建或订阅 Run 的接口里返回）
  - `Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no`（流式接口）
- `cwd` 字段可选，用于指定任务执行目录。开启 `ENFORCE_WORKSPACE_FOR_SERVER` 后，`cwd` 不能逃逸出每个 session 的 workspace。
- 所有 LLM 调用会按真实 token 数返回 `usage`（多轮内部调用会累加）；只有当上游 LLM 没有返回 usage 时才会缺省。

### 错误格式

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

包括校验失败（422）和未匹配的 404，都会被全局 exception handler 包装成同一 schema。部分错误的 `error` 字段还会带额外上下文键：

| HTTP | `error.code` | 含义 | 额外字段 |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | 请求体非法 / 缺字段 | — |
| 400 | `workspace_violation` | `cwd` 逃出允许范围 | — |
| 404 | `run_not_found` / `response_not_found` | 资源不存在 | — |
| 404 | `""` | 通用 not found（如 session） | — |
| 409 | `session_busy` | 该 session 上一轮还在运行 | `status` |
| 409 | `no_regeneratable_answer` | 该 session 没有可重生成的答案 | — |
| 413 | `request_too_large` | 请求体超过 `MAX_REQUEST_BODY_BYTES` | — |
| 422 | `validation_error` | FastAPI 路径 / 查询参数校验失败 | — |
| 429 | `capacity_exceeded` | 全局或单用户并发 run 超额 | `scope`、`limit` |

## 推荐调用方式

按使用场景选择一套接口：

| 场景 | 推荐接口 |
| --- | --- |
| 只需要类似 OpenAI Chat Completions 的对话返回 | `/v1/chat/completions` |
| 需要结构化输出，包含工具调用和工具结果 | `/v1/responses` |
| 需要完整 Agent 生命周期事件，用于自定义前端展示 | `/v1/runs` + `/v1/runs/{run_id}/events` |

### Responses vs Runs

三套接口背后跑的是同一个 Agent，区别在于"暴露什么 + 怎么塑形"：

| 维度 | `/v1/responses` | `/v1/runs` |
| --- | --- | --- |
| 协议定位 | 模仿 OpenAI Responses API | 自定义 Agent 生命周期 |
| 流式事件粒度 | `response.output_text.delta` / `response.output_item.added` / `response.completed` | `reasoning.*` / `tool.delta` / `tool.started` / `tool.updated` / `tool.completed` / `message.delta` / `ask_user` / `run.completed` |
| Agent 思考步骤 | 不暴露 | `reasoning.*` 把每个 LLM 步骤拆出来，`step_id` 把工具调用挂到对应步骤 |
| 工具调用呈现 | 粗：`function_call` + `function_call_output` 两个 item | 细：参数边产生边推、`in_progress`/`updated`/`completed` 状态 |
| `ask_user` 中断 | 不支持 | 支持；下一轮 `POST /v1/runs` 的 `input` 自动作为回答 |
| 断点续传 | 流断了重发 | `cursor` / `Last-Event-ID` 从中间恢复 |
| 多轮串联方式 | `previous_response_id` / `conversation` / `session_id` / `conversation_history` 四选一 | `session_id` 一种 |
| 历史制品 | `store=true` 默认保存，可 `GET /v1/responses/{id}` 取回 / `DELETE` 删除 | 不作为可查阅制品（只能 `GET /v1/runs/{id}` 查状态） |
| 客户端兼容性 | OpenAI Responses SDK 直接可用 | 需要自己写事件解析 |

怎么选：

- **`/v1/responses`**：客户端已经在用 OpenAI SDK；只关心最终结果不要中间过程；想拿历史 response 作可寻址资源；业务侧已有 conversation 标识想直接串轮次。
- **`/v1/runs`**：在做 Agent 展示型 UI，要画"思考 → 调工具 → 工具结果 → 继续思考"的过程条；需要 `ask_user` 这种等用户补充输入的语义；要稳健的断点续传；要把工具参数实时 stream 出来。

一句话总结：**Responses = 最终制品 + OpenAI 兼容**；**Runs = 全过程事件流 + Agent UI 友好**。

## 1. Chat Completions

接口：

```http
POST /v1/chat/completions
```

### 非流式请求

```bash
curl --location "$BASE_URL/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    "stream": false
  }'
```

返回 OpenAI Chat Completions 兼容结构：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1778560000,
  "model": "pairag-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好，我是一个可以使用工具完成任务的 Agent。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
  "metadata": {"session_id": "session_xxx"}
}
```

字段说明：

- `usage` 是本次请求里 Agent 所有内部 LLM 调用累加出来的真实 token 数；调用上游失败或上游 LLM 没返回 usage 时各字段为 0。
- `metadata.session_id` 是本次实际写入历史的 session；客户端透传 `X-Session-Id` 时会一致返回，未传则是服务端新建的 session。

### 流式请求

```bash
curl --no-buffer --location "$BASE_URL/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --header 'X-Session-Id: demo-session-001' \
  --data '{
    "model": "pairag-agent",
    "messages": [
      {"role": "user", "content": "检查当前目录有哪些文件，并简单总结"}
    ],
    "stream": true
  }'
```

流式返回遵循 OpenAI Chat Completions SSE 的文本增量格式：

```text
data: {"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"你好"},"index":0,"finish_reason":null}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":123,"completion_tokens":45,"total_tokens":168}}

data: [DONE]
```

说明：

- `/v1/chat/completions` 是面向通用 OpenAI 客户端的纯文本兼容入口：流式只输出 `delta.content`，不输出 `delta.tool_calls`，也不夹带任何自定义 SSE event。
- 最后一条 `chat.completion.chunk` 会在 chunk 顶层带上 `usage`（行为对齐 OpenAI `stream_options.include_usage=true`）。
- 需要在客户端看到工具调用、工具结果或 Agent 思考步骤，请改用 `/v1/responses`（结构化）或 `/v1/runs/{run_id}/events`（完整生命周期事件）。

## 2. Responses API

接口：

```http
POST /v1/responses
GET /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
```

适合需要结构化结果的调用方。返回中会包含最终消息、工具调用、工具输出。

### 非流式请求

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "input": "你好，请用一句话介绍你自己",
    "stream": false
  }'
```

返回示例：

```json
{
  "id": "resp_xxx",
  "object": "response",
  "created_at": 1778560000,
  "status": "completed",
  "model": "pairag-agent",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "你好，我是一个可以使用工具完成任务的 Agent。"
        }
      ]
    }
  ],
  "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168}
}
```

`usage` 是本次 response 内 Agent 所有 LLM 调用累加得到的真实 token 数；如果调用最终失败、被取消，或调用上游 LLM 没有返回 usage，则可能为 `null`。

包含工具调用时，`output` 可能包含：

```json
[
  {
    "type": "function_call",
    "call_id": "call_1_0",
    "name": "exec_command",
    "arguments": "{\"cmd\":\"ls\"}"
  },
  {
    "type": "function_call_output",
    "call_id": "call_1_0",
    "output": "README.md\nbackend\nfrontends\n"
  },
  {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "当前目录包含 README、后端和前端代码。"}]
  }
]
```

### 流式请求

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "input": "检查当前目录有哪些文件，并简单总结",
    "stream": true
  }'
```

流式事件包括：

| 事件 | 含义 |
| --- | --- |
| `response.created` | Response 已创建 |
| `response.output_item.added` | 新增输出项，例如工具调用或消息 |
| `response.output_text.delta` | 最终文本增量 |
| `response.output_text.done` | 最终文本结束 |
| `response.output_item.done` | 某个输出项结束 |
| `response.completed` | Response 完成 |
| `response.failed` | Response 失败 |

示例：

```text
event: response.created
data: {"type":"response.created","id":"resp_xxx","status":"in_progress"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"你好"}

event: response.completed
data: {"type":"response.completed","id":"resp_xxx","status":"completed","output":[...]}

data: [DONE]
```

### 多轮和会话关联

Responses API 支持以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `input` | string / array | 当前输入，必填 |
| `model` | string | 模型 id：传入则透传给上游 LLM。不传则使用服务端当前生效模型（`runtime_config` 运行时值或环境变量 `MODEL`，详见下文 `/v1/models` 切换接口） |
| `instructions` | string | 本轮系统级说明 |
| `previous_response_id` | string | 继续某个历史 response |
| `conversation` | string | 业务方自定义会话标识 |
| `conversation_history` | array | 调用方显式传入的历史消息（无状态模式） |
| `session_id` | string | 复用服务端会话 |
| `stream` | boolean | 是否流式返回，默认 `false` |
| `store` | boolean | 是否保存 response（用于后续 `GET` / `previous_response_id`），默认 `true` |
| `cwd` | string | 任务执行目录 |

#### 方式一：使用 `previous_response_id`

适合调用方不想额外维护服务端 session，只想基于上一轮 response 继续对话的场景。

第一轮：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "input": "你好，你是谁？",
    "stream": false
  }'
```

从返回中记录 `id`：

```json
{
  "id": "resp_xxx",
  "object": "response",
  "status": "completed"
}
```

第二轮带上 `previous_response_id`：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "previous_response_id": "resp_xxx",
    "input": "那你能帮我做什么？",
    "stream": false
  }'
```

服务会复用上一轮 response 绑定的会话和历史上下文。

#### 方式二：使用 `session_id`

适合调用方想明确控制服务端会话生命周期的场景。

先创建 session：

```bash
SESSION_ID=$(curl -s -X POST "$BASE_URL/v1/sessions" \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
```

第一轮：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data "{
    \"model\": \"pairag-agent\",
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"你好，你是谁？\",
    \"stream\": false
  }"
```

第二轮继续传同一个 `session_id`：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data "{
    \"model\": \"pairag-agent\",
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"继续刚才的话题\",
    \"stream\": false
  }"
```

#### 方式三：使用 `conversation`

适合调用方已经有自己的业务会话 ID，希望用业务 ID 串起多轮 response 的场景。

第一轮：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "conversation": "user-123-ticket-456",
    "input": "第一轮问题",
    "stream": false
  }'
```

第二轮继续传同一个 `conversation`：

```bash
curl --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "conversation": "user-123-ticket-456",
    "input": "第二轮问题",
    "stream": false
  }'
```

推荐选择：

| 场景 | 推荐方式 |
| --- | --- |
| 后端集成，想最少维护状态 | `previous_response_id` |
| 调用方已有自己的业务会话 ID | `conversation` |
| 调用方要明确创建、查询、删除服务端会话 | `session_id` |

#### 优先级与互斥关系

一次请求可以同时携带多个上下文字段，服务端按下面的顺序解析：

1. **`previous_response_id`**：若给出，必须能加载到对应 response，否则返回 404 `response_not_found`。该 response 的 session、对话历史、`instructions`、`conversation` 会作为本轮默认值。
2. **`conversation`**（仅在没有 `previous_response_id` 时生效）：查找该 `conversation` 标识下最新一条 response 作为隐式 previous。
3. **`session_id`**：始终是显式优先；若没传，则继承自第 1/2 步推导出的 previous response。
4. **`conversation_history`**：无状态模式。若传了，直接覆盖从 1/2 推出的历史；适合调用方完全在客户端维护对话上下文，不想依赖服务端 session。
5. **`instructions`**：本轮显式 `instructions` 覆盖 previous response 上的 `instructions`。

读取 response：

```bash
curl --location "$BASE_URL/v1/responses/resp_xxx"
```

删除 response：

```bash
curl --request DELETE --location "$BASE_URL/v1/responses/resp_xxx"
```

## 3. Runs API

Runs API 是完整结构化事件接口，适合自定义前端或需要展示 Agent 思考步骤、工具状态、最终输出的调用方。

### 创建 Run

```bash
curl --location "$BASE_URL/v1/runs" \
  --header 'Content-Type: application/json' \
  --data '{
    "input": "检查当前目录有哪些文件，并简单总结"
  }'
```

返回：

```json
{
  "id": "run_xxx",
  "object": "agent.run",
  "run_id": "run_xxx",
  "session_id": "session_xxx",
  "status": "started",
  "cursor": "0-0"
}
```

`cursor` 是订阅事件用的位置标识。后续调用 `GET /v1/runs/{run_id}/events?last_event_id=<cursor>`（或带头 `Last-Event-ID`）可从该位置断点续传。

如果不想走两段式（POST 创建 → GET 订阅），可以传 `stream=true` 一次性拿到 SSE 流，事件 schema 与 `GET /v1/runs/{run_id}/events` 完全一致：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs" \
  --header 'Content-Type: application/json' \
  --data '{
    "input": "检查当前目录有哪些文件，并简单总结",
    "stream": true
  }'
```

也可以先创建 session，再创建 run：

```bash
SESSION_ID=$(curl -s -X POST "$BASE_URL/v1/sessions" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl --location "$BASE_URL/v1/runs" \
  --header 'Content-Type: application/json' \
  --data "{\"session_id\":\"${SESSION_ID}\",\"input\":\"你好\"}"
```

### 多轮和会话关联

Runs API 通过 `session_id` 维持多轮上下文。每一轮用户输入都会创建一个新的 `run_id`，但只要使用同一个 `session_id`，Agent 就会复用该 session 内的历史消息和状态。

推荐流程：

1. 先创建 session。
2. 每一轮调用 `/v1/runs` 时传同一个 `session_id`。
3. 每一轮根据返回的新 `run_id` 订阅 `/v1/runs/{run_id}/events`。

创建 session：

```bash
SESSION_ID=$(curl -s -X POST "$BASE_URL/v1/sessions" \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
```

第一轮创建 run：

```bash
RUN_ID_1=$(curl -s --location "$BASE_URL/v1/runs" \
  --header 'Content-Type: application/json' \
  --data "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"你好，你是谁？\"
  }" | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
```

订阅第一轮事件：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/${RUN_ID_1}/events"
```

第二轮继续使用同一个 `session_id` 创建新的 run：

```bash
RUN_ID_2=$(curl -s --location "$BASE_URL/v1/runs" \
  --header 'Content-Type: application/json' \
  --data "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"那你能帮我做什么？\"
  }" | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
```

订阅第二轮事件：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/${RUN_ID_2}/events"
```

说明：

- `session_id` 表示长期会话，负责串起多轮上下文。
- `run_id` 表示某一轮任务执行，每轮输入都会生成新的 `run_id`。
- 如果上一轮还在运行，继续向同一个 `session_id` 创建 run 可能返回 `session_busy`。
- 如果事件中出现 `ask_user`，表示当前 run 正在等待用户补充信息；此时继续向同一个 `session_id` 调用 `/v1/runs`，请求体中的 `input` 会作为用户补充答案提交给当前 run。

### 订阅 Run 事件

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/run_xxx/events"
```

事件通过 SSE `data:` 输出，JSON 内的 `event` 字段表示事件类型。所有事件都带 `run_id` 和 `timestamp`（float 秒）；多数事件还会带 `step_id`（如 `model-1`）把工具调用挂到对应的 Agent 步骤：

| `event` | 含义 | 关键字段 |
| --- | --- | --- |
| `message.delta` | 最终回答文本增量 | `delta` |
| `reasoning.started` | Agent 步骤开始 | `step_id`、`title`、`hidden` |
| `reasoning.available` | Agent 步骤内容更新 | `step_id`、`text`、`replace` |
| `reasoning.completed` | Agent 步骤结束 | `step_id`、`status`、`text` |
| `tool.delta` | 工具调用参数增量（流式 tool args） | `tool_call_id`、`tool`、`arguments_delta`、`arguments_text` |
| `tool.started` | 工具开始执行 | `tool_call_id`、`tool`、`input`、`kind` |
| `tool.updated` | 工具状态更新（运行中） | `tool_call_id`、`status`、`content`、`data` |
| `tool.completed` | 工具执行结束 | `tool_call_id`、`status`（`completed` / `failed`）、`content`、`data` |
| `ask_user` | Agent 需要用户补充信息 | `question`、`candidates` |
| `run.completed` | Run 完成 | `output`、`usage` |
| `run.failed` | Run 失败 | `error` |

`tool_call_id` 格式：`call_{turn}_{index}`（OpenAI 风格 `call_` 前缀）。

示例：

```text
data: {"event":"reasoning.started","run_id":"run_xxx","timestamp":1778640000.1,"step_id":"model-1","title":"Agent step","status":"in_progress","hidden":false}

data: {"event":"tool.delta","run_id":"run_xxx","timestamp":1778640000.5,"tool_call_id":"call_1_0","step_id":"model-1","tool":"exec_command","arguments_delta":"{\"cm","arguments_text":"{\"cm"}

data: {"event":"tool.started","run_id":"run_xxx","timestamp":1778640000.7,"tool_call_id":"call_1_0","step_id":"model-1","tool":"exec_command","input":{"cmd":"ls"}}

data: {"event":"tool.completed","run_id":"run_xxx","timestamp":1778640001.2,"tool_call_id":"call_1_0","tool":"exec_command","status":"completed","content":"README.md\nbackend\n"}

data: {"event":"message.delta","run_id":"run_xxx","timestamp":1778640001.5,"delta":"当前目录包含 README 和 backend。"}

data: {"event":"run.completed","run_id":"run_xxx","timestamp":1778640001.8,"output":"当前目录包含 README 和 backend。","usage":{"prompt_tokens":80,"completion_tokens":25,"total_tokens":105}}
```

断线重连时可传入上次事件 ID：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/run_xxx/events?last_event_id=1700000000000-0"
```

也可使用请求头：

```http
Last-Event-ID: 1700000000000-0
```

### 查询 Run 状态

```bash
curl --location "$BASE_URL/v1/runs/run_xxx"
```

返回：

```json
{
  "id": "run_xxx",
  "object": "agent.run",
  "run_id": "run_xxx",
  "session_id": "session_xxx",
  "status": "completed",
  "mode": "events",
  "error": "",
  "created_at": "2026-05-13T10:56:54.056735",
  "updated_at": "2026-05-13T10:56:55.780018",
  "started_at": "2026-05-13T10:56:54.056735",
  "finished_at": "2026-05-13T10:56:55.780018",
  "last_event_id": "1778640002000-0"
}
```

`status` 取值：`started` / `running` / `completed` / `failed` / `cancelled` / `waiting_user`。时间字段是 ISO-8601 字符串；`error` 仅在 `failed` 时有内容；`last_event_id` 是 Redis Stream cursor 形式，可作为 `Last-Event-ID` 用于断点续传。

### 停止 Run

```bash
curl --request POST --location "$BASE_URL/v1/runs/run_xxx/stop"
```

返回 `{"run_id": "run_xxx", "status": "stopping"}`。等价于 `POST /v1/sessions/{session_id}/cancel`（取消整条 session）；当前同一 session 只有一个 active run，二者效果一致。

## 4. Sessions API

Sessions 用于服务端多轮会话管理。Chat Completions 通过 `X-Session-Id` 复用会话；Runs 和 Responses 通过请求体里的 `session_id` 复用会话。

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

时间字段是 ISO-8601 字符串；`status` 取值 `idle` / `running` / `waiting_user` / `completed` / `failed` / `cancelled`。

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
返回 `{"cancelled":true,"session_id":"session_xxx"}`。与 `POST /v1/runs/{run_id}/stop` 等价。

### 重新生成上一轮答案

```bash
curl --request POST --location "$BASE_URL/v1/sessions/session_xxx/regenerate" \
  --header 'Content-Type: application/json' \
  --data '{}'
```

返回新的 run（与 `POST /v1/runs` 同结构，多 `regenerated_from_run_id` 字段），需要继续订阅：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/run_xxx/events"
```

若该 session 没有可重生成的答案，返回 409 `no_regeneratable_answer`。

### 错误码补充

| HTTP | `error.code` | 触发条件 |
| --- | --- | --- |
| 404 | — | session 不存在 |
| 409 | `session_busy` | 该 session 上一轮还在运行，无法启动新 run |
| 409 | `no_regeneratable_answer` | 该 session 没有可重生成的答案 |
| 429 | `capacity_exceeded` | 全局或单用户并发 run 超 `MAX_GLOBAL_RUNS` / `MAX_USER_RUNS` |

## 5. 其他接口

### 健康检查

```bash
curl --location "$BASE_URL/health"          # {"status":"ok"}
curl --location "$BASE_URL/health/detailed" # 见下方示例
```

`/health/detailed` 返回示例：

```json
{
  "status": "ok",
  "runner_backend": "thread",
  "model": "qwen-plus",
  "checks": {
    "api": "ok",
    "sqlite": "ok",
    "redis": "disabled",
    "runner": "thread"
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

`POST /v1/models/active` 修改全局生效模型。改动**立即对所有新建对话生效**（已经在跑的 stream 沿用原模型），并写入 `memory/active_model.json` 跨进程 / 跨重启持久化。HTTP server / Celery worker / ACP server 共享同一份。

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

### Skills 列表

```bash
curl --location "$BASE_URL/v1/skills"
```
列出 `skills/` 目录下可被 `use_skill` 工具调用的技能。

## 6. 部署后 Smoke Test

仓库提供了一个纯 Python 标准库脚本，用来验证部署后的三类 API 是否可用：

- `/v1/chat/completions`
- `/v1/responses`
- `/v1/runs` + `/v1/runs/{run_id}/events`

默认 sample query：

```text
你好，请用一句话介绍你自己，并说明你可以通过 API 被调用。
```

Runs API 默认 sample query：

```text
请确认 /v1/runs 接口可以正常执行，并用一句话说明 Runs API 的作用。
```

运行方式：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --model "pairag-agent"
```

如果部署网关需要鉴权：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --model "pairag-agent" \
  --auth "Bearer xxx"
```

如果鉴权头不是 `Authorization`，可以使用自定义 header：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --header "Authorization: your-token" \
  --header "X-Request-Id: smoke-test-001"
```

只测试某一类 API：

```bash
python scripts/api_smoke.py --base-url "$BASE_URL" --only chat
python scripts/api_smoke.py --base-url "$BASE_URL" --only responses
python scripts/api_smoke.py --base-url "$BASE_URL" --only runs
```

自定义 sample query：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --query "你好，请说明你是什么服务" \
  --run-query "请确认 Runs API 可以正常返回事件"
```

测试 Responses API 的 `previous_response_id` 多轮：

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --only responses \
  --multi-turn
```

脚本成功时会打印每类 API 的状态码、`session_id`、`run_id` 或 `response_id`，以及解析出的最终答案。

## 7. 常见问题

### 为什么 Chat Completions 流式没有 `delta.tool_calls`？

Agent 的工具调用是服务端自动完成的，不要求客户端回传工具结果。`/v1/chat/completions` 定位是给通用 OpenAI 客户端使用的纯文本兼容入口，所以不输出 `delta.tool_calls`，也不带任何自定义事件。需要看到工具调用、工具结果或 Agent 思考步骤，请改用 `/v1/responses` 或 `/v1/runs/{run_id}/events`。

### 什么时候用 `session_id`？

需要多轮上下文时使用。调用方可以：

- Chat Completions：传 `X-Session-Id`
- Responses：传 `session_id` 或 `previous_response_id`
- Runs：创建 run 时传 `session_id`

### 非流式接口会等多久？

非流式接口会等待本次 Agent 任务完成后再返回。复杂任务建议使用流式接口，避免网关或客户端超时。

### SSE 中的 keepalive 是什么？

长时间没有新事件时，服务可能发送注释行：

```text
: keepalive
```

客户端应忽略这类注释行。

### 怎么拿到真实 token 使用量？

- Chat Completions（非流式）：响应顶层 `usage`。
- Chat Completions（流式）：最后一条 `chat.completion.chunk` 顶层 `usage`。
- Responses（非流式）：响应顶层 `usage`。
- Responses（流式）：最后一条 `response.completed` 事件里的 `usage`。
- Runs：`run.completed` 事件里的 `usage`。

`usage` 是本次请求 Agent 所有内部 LLM 调用的累加值。任务失败、被取消，或上游 LLM 没回 usage 时，各字段为 0；Responses 非流式且无任何 token 数据时 `usage` 可能为 `null`。

### 单一 session 能并发跑多个 run 吗？

不能。同一 `session_id` 同一时刻只允许一个 active run；上一轮还在跑时再发 `POST /v1/runs` 会返回 409 `session_busy`。需要并发请走不同的 session。
