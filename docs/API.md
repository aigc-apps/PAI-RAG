# PAI-RAG API

当前公开模型调用支持两条正式 API：

- `/v1/responses`：推荐入口，暴露结构化 SSE、工具调用、HITL 和 response 存取。
- `/v1/chat/completions`：通用 OpenAI Chat Completions 客户端入口。

旧的 `/v1/runs` 已删除，现返回 404。

## 基础约定

- Base URL 示例：`http://127.0.0.1:8000`
- `/v1/responses` 只支持 SSE 流式：`stream=true`；`stream=false` 返回 `400 unsupported_mode`
- `/v1/chat/completions` 支持标准 `messages`，可用 `stream=true` 或非流式
- `session_id`、`conversation_history`、Responses 请求里的 `messages`、`X-Session-Id` 请求头均已删除，传入会返回 400
- API 层成功只代表请求被 Agent 接受并跑完；业务校验是否成功要看最终文本、`response.status` 与 `error`

## Health

```bash
curl "$BASE_URL/health"
curl "$BASE_URL/health/detailed"
```

## Models

```bash
curl "$BASE_URL/v1/models"

curl -X POST "$BASE_URL/v1/models/active" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-plus"}'
```

`POST /v1/models/active` 修改全局默认模型。`POST /v1/responses` 与
`POST /v1/chat/completions` 的 `model` 字段只覆盖当次请求，不写入全局。

## Chat Completions

```text
POST /v1/chat/completions
```

用于通用 OpenAI Chat Completions 客户端。多轮上下文由客户端放在
`messages` 数组中；服务端不再读取 `X-Session-Id`。

### 非流式

```bash
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-plus",
    "messages": [
      {"role": "user", "content": "你好，用一句话介绍你自己"}
    ],
    "stream": false
  }'
```

### 流式

```bash
curl --no-buffer -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-plus",
    "messages": [
      {"role": "user", "content": "你好，用一句话介绍你自己"}
    ],
    "stream": true
  }'
```

Chat Completions 默认只返回最终文本增量。需要完整工具调用、工具结果、
HITL `requires_action` 和 response 持久化时，请使用 `/v1/responses`。

## Responses

```text
POST   /v1/responses
GET    /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
POST   /v1/responses/{response_id}/cancel
```

### 单轮调用

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-plus",
    "input": "你好，用一句话介绍你自己",
    "stream": true
  }'
```

SSE 至少包含 `response.created`，并以 `response.completed`、
`response.failed` 或 `response.incomplete` 作为终态，最后一行为
`data: [DONE]`。

### 多轮调用

推荐二选一：

1. 使用 `conversation` 作为业务侧会话串号。
2. 使用上一轮 `response.completed.id` 作为下一轮的 `previous_response_id`。

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "biz-conv-001",
    "input": "记住数字 7，然后回复好的",
    "stream": true
  }'

curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "biz-conv-001",
    "input": "我刚才让你记住的数字是什么？",
    "stream": true
  }'
```

Web 前端会先创建 `/v1/sessions`，再把返回的 `session_id` 放入
`conversation` 字段；外部调用方可以直接使用自己的业务 conversation id。

### HITL 续答

启动请求需要显式允许 HITL：

```json
{
  "input": "需要用户确认时先暂停",
  "allow_hitl": true,
  "stream": true
}
```

如果终态是 `response.requires_action` / `response.incomplete`，客户端保存
`response_id` 与 `tool_calls[].id`，再用 `previous_response_id` +
`function_call_output` 续答：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "previous_response_id": "resp_xxx",
    "input": [{
      "type": "function_call_output",
      "call_id": "call_xxx",
      "output": "用户选择 A"
    }],
    "stream": true
  }'
```

## Sessions

Sessions 是当前 Web 前端的会话管理接口，不是公开模型调用协议。
模型调用仍应走 `/v1/responses`，并使用 `conversation` 维持多轮。

```text
GET    /v1/sessions
POST   /v1/sessions
GET    /v1/sessions/{session_id}
POST   /v1/sessions/{session_id}/regenerate
POST   /v1/sessions/{session_id}/cancel
DELETE /v1/sessions/{session_id}
```

`POST /v1/sessions/{session_id}/regenerate` 返回 SSE，事件格式与
`POST /v1/responses` 相同。

## 已删除接口

以下旧接口不再保留兼容层：

```text
POST /v1/runs
GET  /v1/runs/{run_id}
GET  /v1/runs/{run_id}/events
POST /v1/runs/{run_id}/stop
```

## 业务配置校验示例

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "engine-config-check-embedding-config",
    "input": "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。",
    "stream": true
  }'
```

如果配置存在业务错误，API 仍可能以 `response.completed` 正常结束；最终文本会说明校验失败与错误摘要。

## Smoke

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --multi-turn
```

该脚本会验证 `/v1/chat/completions`、`/v1/responses` 正向调用，并确认旧的
`/v1/runs` 返回 404。
