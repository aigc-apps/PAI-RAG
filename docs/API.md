# MiniAgent 服务 API 文档

本文档面向服务部署后的调用方，说明如何通过 HTTP API 调用 Agent。示例中的 `BASE_URL` 请替换为实际部署地址。

```bash
export BASE_URL="http://127.0.0.1:8000"
```

如果部署平台在网关层要求鉴权，请按平台要求附加 `Authorization` 等请求头。当前后端服务本身不校验鉴权头。

## 通用约定

- 请求体使用 JSON：`Content-Type: application/json`
- 流式接口使用 SSE：`Content-Type: text/event-stream`
- 服务端会在响应头中尽量返回：
  - `X-Session-Id`：当前会话 ID
  - `X-Run-Id`：当前运行 ID
- `cwd` 可选，用于指定任务执行目录。受 workspace 约束时，`cwd` 不能逃逸出允许的 workspace。
- 普通错误格式：

```json
{
  "error": {
    "message": "Run not found: run_xxx",
    "type": "invalid_request_error",
    "code": "run_not_found"
  }
}
```

## 推荐调用方式

按使用场景选择一套接口：

| 场景 | 推荐接口 |
| --- | --- |
| 只需要类似 OpenAI Chat Completions 的对话返回 | `/v1/chat/completions` |
| 需要结构化输出，包含工具调用和工具结果 | `/v1/responses` |
| 需要完整 Agent 生命周期事件，用于自定义前端展示 | `/v1/runs` + `/v1/runs/{run_id}/events` |

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
  "id": "chatcmpl_xxx",
  "object": "chat.completion",
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
  ]
}
```

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
data: {"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"你好"},"index":0}]}

data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

当 Agent 执行工具时，服务会额外输出自定义 SSE 事件：

```text
event: pai.tool.progress
data: {"object":"pai.tool.progress","run_id":"run_xxx","tool":"exec_command","status":"running","preview":"Run command"}
```

说明：

- `stream=true` 时不会伪造 OpenAI `delta.tool_calls`。
- 文本答案仍通过 OpenAI Chat 的 `delta.content` 输出。
- 工具进度通过 `event: pai.tool.progress` 单独输出，客户端可选择忽略。

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
  "usage": null
}
```

包含工具调用时，`output` 可能包含：

```json
[
  {
    "type": "function_call",
    "call_id": "tool-1-0",
    "name": "exec_command",
    "arguments": "{\"cmd\":\"ls\"}"
  },
  {
    "type": "function_call_output",
    "call_id": "tool-1-0",
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
| `instructions` | string | 本轮系统级说明 |
| `previous_response_id` | string | 继续某个历史 response |
| `conversation` / `conversation_id` | string | 业务方自定义会话标识 |
| `conversation_history` | array | 调用方显式传入的历史消息 |
| `session_id` | string | 复用服务端会话 |
| `stream` | boolean | 是否流式返回 |
| `store` | boolean | 是否保存 response，默认 `true` |
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
  "stream_from": "0-0"
}
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

事件通过 SSE `data:` 输出，JSON 内的 `event` 字段表示事件类型：

| `event` | 含义 |
| --- | --- |
| `message.delta` | 最终回答文本增量 |
| `reasoning.started` | Agent 步骤开始 |
| `reasoning.available` | Agent 步骤内容更新 |
| `reasoning.completed` | Agent 步骤结束 |
| `tool.delta` | 工具调用参数增量 |
| `tool.started` | 工具开始执行 |
| `tool.updated` | 工具状态更新 |
| `tool.completed` | 工具执行结束 |
| `ask_user` | Agent 需要用户补充信息 |
| `run.completed` | Run 完成 |
| `run.failed` | Run 失败 |

示例：

```text
data: {"event":"reasoning.started","run_id":"run_xxx","step_id":"model-1","title":"Agent step"}

data: {"event":"tool.started","run_id":"run_xxx","tool":"exec_command","tool_call_id":"tool-1-0"}

data: {"event":"message.delta","run_id":"run_xxx","delta":"当前目录包含 README 和 backend。"}

data: {"event":"run.completed","run_id":"run_xxx","output":"当前目录包含 README 和 backend。"}
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

### 停止 Run

```bash
curl --request POST --location "$BASE_URL/v1/runs/run_xxx/stop"
```

## 4. Sessions API

Sessions 用于服务端多轮会话管理。Chat Completions 可通过 `X-Session-Id` 复用会话；Runs 和 Responses 可通过 `session_id` 复用会话。

### 创建 Session

```bash
curl --request POST --location "$BASE_URL/v1/sessions" \
  --header 'Content-Type: application/json' \
  --data '{}'
```

返回：

```json
{
  "session_id": "session_xxx",
  "title": "New Task",
  "status": "idle",
  "messages": []
}
```

### 查询、列表、删除

```bash
curl --location "$BASE_URL/v1/sessions"
curl --location "$BASE_URL/v1/sessions/session_xxx"
curl --request DELETE --location "$BASE_URL/v1/sessions/session_xxx"
```

### 取消当前 Session

```bash
curl --request POST --location "$BASE_URL/v1/sessions/session_xxx/cancel"
```

### 重新生成上一轮答案

```bash
curl --request POST --location "$BASE_URL/v1/sessions/session_xxx/regenerate" \
  --header 'Content-Type: application/json' \
  --data '{}'
```

返回新的 run，需要继续订阅：

```bash
curl --no-buffer --location "$BASE_URL/v1/runs/run_xxx/events"
```

## 5. 其他接口

### 健康检查

```bash
curl --location "$BASE_URL/health"
curl --location "$BASE_URL/health/detailed"
```

### 模型列表

```bash
curl --location "$BASE_URL/v1/models"
```

### Skills 列表

```bash
curl --location "$BASE_URL/v1/skills"
```

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

Agent 的工具调用是服务端执行的，不要求客户端回传工具结果。因此 Chat Completions 流式返回保持 OpenAI 文本增量兼容，工具进度用自定义事件 `pai.tool.progress` 单独发送。

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
