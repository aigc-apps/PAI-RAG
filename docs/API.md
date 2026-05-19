# PAI-RAG API

推荐使用 `/v1/responses`，它返回 SSE 流，既能拿最终回答，也能拿中间思考、工具调用和工具结果。

## 基础约定

- Base URL 示例：`http://127.0.0.1:8683`
- 请求体：JSON，需带 `Content-Type: application/json`
- `/v1/responses` 只支持流式：必须传 `"stream": true`
- 最终成功终态：`response.completed`
- 失败终态：`response.failed`
- 需要人工续答：`response.requires_action`（已实现，暂不需要）
- EAS服务需要鉴权，请在header中传入 `Authorization`

## 最小调用

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{"input":"请帮我创建一个文件，a.txt","stream":true}'
```

返回是 `text/event-stream`。业务方按 SSE 读取 `event:` 和 `data:`。

## 请求参数

`POST /v1/responses`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `input` | string / object / array | 是 | 用户输入。普通业务建议直接传 string。 |
| `stream` | boolean | 是 | 必须为 `true`。 |
| `model` | string | 否 | 覆盖本次请求使用的模型。 |
| `conversation` | string | 否 | 业务会话 ID；同一值会串联上下文。 |
| `previous_response_id` | string | 否 | 基于上一轮 response 继续对话或 HITL 续答。 |
| `allow_hitl` | boolean | 否 | 是否允许任务暂停等待业务方补充输入。默认 `false`。 |
| `store` | boolean | 否 | 是否保存 response，默认 `true`。 |
| `cwd` | string | 否 | 工具执行目录；必须在服务允许的 workspace 内。 |
| `aliyun_credentials` | object | 否 | 本次请求使用的 Aliyun AK/SK。 |

Aliyun 凭证格式：

```json
{
  "aliyun_credentials": {
    "access_key_id": "<access-key-id>",
    "access_key_secret": "<access-key-secret>",
    "region_id": "cn-beijing"
  }
}
```

不要把 AK/SK 写进 `input`。

请求带 `aliyun_credentials` 时，服务会先用这组 AK/SK 执行 `aliyun configure set` 写入请求级临时 profile，并在本次工具执行中优先使用该 profile。请求结束后会清理这个临时 profile。

如果工具执行时发现请求传入的 AK/SK 无效，Agent 会停止任务，并在最终结果中说明 AK/SK 无效；不会改用机器默认凭证继续执行。

## 多轮对话

多轮对话仍然是流式调用。推荐使用 `conversation` 作为业务会话 ID；同一个 `conversation` 会自动接上上一轮上下文。多轮依赖服务端保存上一轮 response，请保持 `store=true`，也就是默认值。

第一轮：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "conversation": "biz-conv-001",
    "input": "记住：订单号是 A1001",
    "stream": true
  }'
```

第二轮：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "conversation": "biz-conv-001",
    "input": "我刚才说的订单号是什么？",
    "stream": true
  }'
```

也可以不用 `conversation`，而是在上一轮 `response.completed` 的 `id` 里拿到 `resp_xxx`，下一轮传 `previous_response_id`：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "previous_response_id": "resp_xxx",
    "input": "继续上一个问题，再补充说明风险点",
    "stream": true
  }'
```

两种方式二选一即可。客户端已有自己的会话 ID 时，建议用 `conversation`。若没有，可以直接用`previous_response_id`。

## 读取 SSE 结果

`/v1/responses` 只有流式返回。客户端读取 SSE 时，每个业务事件由两部分组成：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"文件","sequence_number":8}
```

常用字段：

| 字段 | 出现位置 | 说明 |
| --- | --- | --- |
| `event` | SSE 行 | 事件名，例如 `response.output_text.delta`。 |
| `data.type` | JSON | 通常等于 `event`，便于只解析 JSON 的客户端判断事件类型。 |
| `data.sequence_number` | JSON | 事件序号，按服务端发送顺序递增。 |
| `data.response_id` / `data.id` | JSON | response ID。`response.completed.id` 可用于下一轮 `previous_response_id`。 |
| `data.delta` | JSON | 增量文本，常见于正文、思考、工具参数流式事件。 |
| `data.text` | JSON | 某段文本的完整内容，常见于 `response.output_text.done`。 |
| `data.item` | JSON | 输出项，可能是 `message`、`reasoning`、`function_call`、`function_call_output`。 |
| `data.output` | JSON | 终态完整输出数组，只在 `response.completed`、`response.requires_action` 等终态事件里读取。 |
| `data.error` | JSON | 失败信息，常见于 `response.failed`。 |

按需求读取即可：

| 业务需求 | 读取方式 |
| --- | --- |
| 只拿最终结果 | 忽略中间事件，等待 `response.completed`，从 `data.output` 提取最终正文。 |
| 实时展示正文 | 拼接所有 `response.output_text.delta` 的 `delta`。 |
| 展示思考过程 | 按 `step_id` 拼接 `response.reasoning_text.delta` 的 `delta`。 |
| 展示工具调用 | 读取 `response.output_item.added/done`，当 `item.type="function_call"` 时展示 `item.name`、`item.arguments`。 |
| 展示工具结果 | 读取 `response.output_item.added/done`，当 `item.type="function_call_output"` 时展示 `item.output`。 |
| 判断任务结束 | 看 `response.completed`、`response.failed`、`response.requires_action` 或 `response.incomplete`。不要把 `response.output_text.done` 当作整次请求结束。 |

最终正文提取规则：

1. 优先取 `message.metadata.pai_final_report=true` 的 `output_text`。
2. 如果有工具结果，优先取最后一个 `function_call_output` 后面的 `message`。
3. 否则取普通 `message`。
4. 忽略 `reasoning`，它是过程思考，不是最终正文。

Python 流式示例：

```python
import json
import requests

BASE_URL = "http://127.0.0.1:8683"


def final_text(response):
    output = response.get("output", [])
    last_tool = max(
        (i for i, item in enumerate(output) if item.get("type") == "function_call_output"),
        default=-1,
    )
    reports, after_tools, fallback = [], [], []

    for i, item in enumerate(output):
        if item.get("type") != "message":
            continue
        text = "".join(
            block.get("text", "")
            for block in item.get("content", [])
            if block.get("type") in ("output_text", "text")
        )
        if not text:
            continue
        if (item.get("metadata") or {}).get("pai_final_report"):
            reports.append(text)
        elif last_tool >= 0 and i > last_tool:
            after_tools.append(text)
        else:
            fallback.append(text)

    return "".join(reports or after_tools or fallback)


answer, reasoning, tools = [], {}, []
completed = None

with requests.post(
    f"{BASE_URL}/v1/responses",
    json={"input": "请帮我创建一个文件，a.txt", "stream": True},
    stream=True,
    timeout=300,
) as resp:
    resp.raise_for_status()
    event = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue

        raw = line[len("data:"):].strip()
        if raw == "[DONE]":
            break
        data = json.loads(raw)

        if event == "response.output_text.delta":
            answer.append(data.get("delta", ""))
        elif event == "response.reasoning_text.delta":
            step_id = data.get("step_id", "")
            reasoning[step_id] = reasoning.get(step_id, "") + data.get("delta", "")
        elif event in ("response.output_item.added", "response.output_item.done"):
            item = data.get("item") or {}
            if item.get("type") in ("function_call", "function_call_output"):
                tools.append(item)
        elif event == "response.failed":
            raise RuntimeError((data.get("error") or {}).get("message", "run failed"))
        elif event == "response.requires_action":
            raise RuntimeError("requires_action")
        elif event == "response.completed":
            completed = data

print("实时正文：", "".join(answer))
print("最终正文：", final_text(completed or {}))
print("思考步骤数：", len(reasoning))
print("工具事件数：", len(tools))
```

## SSE 事件

SSE 示例：

```text
event: response.created
data: {"id":"resp_xxx","status":"in_progress","type":"response.created","sequence_number":1}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"文件 a.txt 已创建。","sequence_number":8}

event: response.completed
data: {"id":"resp_xxx","status":"completed","output":[...],"type":"response.completed","sequence_number":9}

data: [DONE]
```

服务端可能发送 keepalive：

```text
: keepalive
```

客户端忽略以 `:` 开头的行。

常见事件：

| 事件 | 说明 |
| --- | --- |
| `response.created` | 任务已创建。保存 `id`，可作为 `previous_response_id`。 |
| `response.reasoning_step.started` | 一个过程步骤开始。 |
| `response.reasoning_text.delta` | 过程思考文本增量，不是最终正文。 |
| `response.reasoning_step.completed` | 一个过程步骤结束。 |
| `response.output_text.delta` | 用户可见最终回答增量。 |
| `response.output_text.done` | 一段用户可见文本完成；它不是整次请求终态。 |
| `response.output_item.added` | 新增输出项，可能是工具调用、工具结果、reasoning、message。 |
| `response.output_item.done` | 输出项完成。 |
| `response.function_call_arguments.delta` | 工具参数增量，通常可忽略。 |
| `response.function_call_arguments.done` | 工具参数完成，通常可忽略。 |
| `response.requires_action` | HITL 暂停，需要业务方续答。 |
| `response.incomplete` | HITL 暂停后的兼容结束事件。 |
| `response.completed` | 成功终态。 |
| `response.failed` | 失败终态。 |
| `[DONE]` | SSE 流结束。 |

所有 JSON 事件都有 `sequence_number`，可用于日志排序和排查。

业务方判断一次请求是否结束，应看 `response.completed`、`response.failed`、`response.requires_action` 或 `response.incomplete`，不要只看 `response.output_text.done`。

## output 结构

`response.completed.output` 是数组，常见 `item.type`：

| type | 说明 | 是否最终正文 |
| --- | --- | --- |
| `message` | 用户可见回答。内容在 `content[].text`。 | 是 |
| `reasoning` | 过程思考。内容在 `content[].text`。 | 否 |
| `function_call` | 工具调用。包含 `name`、`arguments`。 | 否 |
| `function_call_output` | 工具结果。包含 `output`。 | 否 |

最终正文示例：

```json
{
  "type": "message",
  "role": "assistant",
  "status": "completed",
  "content": [
    {"type": "output_text", "text": "文件 a.txt 已创建。"}
  ]
}
```

工具调用示例：

```json
{
  "type": "function_call",
  "call_id": "call_xxx",
  "name": "file_write",
  "arguments": "{\"path\":\"a.txt\",\"content\":\"\"}",
  "status": "completed"
}
```

模型内部标签，例如 `<thinking>`、`<taking>`、`<summary>`，不会作为最终正文输出。思考内容会通过 `response.reasoning_text.delta` 或 `output[].type="reasoning"` 表示。

如果 `message.metadata.pai_final_report=true`，该 message 是服务端最终报告，优先作为最终正文。

## HITL 续答

如果收到 `response.requires_action`，读取：

```text
required_action.submit_tool_outputs.tool_calls[].id
```

然后用 `previous_response_id` 续答：

```bash
curl --no-buffer --location "$BASE_URL/v1/responses" \
  --header 'Content-Type: application/json' \
  --data '{
    "previous_response_id": "resp_xxx",
    "input": [{
      "type": "function_call_output",
      "call_id": "call_xxx",
      "output": "同意执行"
    }],
    "stream": true
  }'
```

## 查询、删除、取消

```bash
# 查询保存的 response
curl "$BASE_URL/v1/responses/resp_xxx"

# 删除保存的 response
curl -X DELETE "$BASE_URL/v1/responses/resp_xxx"

# 取消正在执行或已保存的 response
curl -X POST "$BASE_URL/v1/responses/resp_xxx/cancel"
```

删除成功返回：

```json
{"id": "resp_xxx", "object": "response.deleted", "deleted": true}
```

## 错误码

### HTTP JSON 错误

请求未进入流式执行前失败，会直接返回 JSON：

```json
{
  "error": {
    "message": "Non-streaming /v1/responses is not supported; pass stream=true",
    "type": "invalid_request_error",
    "code": "unsupported_mode"
  }
}
```

常见错误码：

| HTTP | code | 说明 |
| --- | --- | --- |
| 400 | `invalid_request_error` | JSON 非法、字段类型错误、model 非法等。 |
| 400 | `unsupported_mode` | `/v1/responses` 没有传 `stream=true`。 |
| 400 | `unsupported_legacy_field` | 传了已删除字段，如 `session_id`、`messages`。 |
| 400 | `unsupported_legacy_header` | 传了 `X-Session-Id`。 |
| 400 | `invalid_aliyun_credentials` | Aliyun AK/SK 格式错误。 |
| 400 | `invalid_resume` | HITL 续答格式错误。 |
| 400 | `workspace_violation` | `cwd` 或工具路径不在允许 workspace 内。 |
| 413 | `request_too_large` | 请求体过大。 |
| 422 | `validation_error` | FastAPI 参数校验失败。 |
| 404 | `response_not_found` | response 不存在。 |
| 409 | `not_resumable` | response 不是等待续答状态。 |
| 409 | `session_busy` | 同一 session 正在执行。 |
| 429 | `capacity_exceeded` | 服务端并发或容量限制。 |
| 500 | `aliyun_config_error` | 服务端写入 Aliyun 临时配置失败。 |

### 流内错误

任务已开始后失败，HTTP 通常仍是 200，错误在 SSE 中：

```text
event: response.failed
data: {"id":"resp_xxx","status":"failed","error":{"message":"...","code":"..."}}

data: [DONE]
```

常见流内错误：

| code | 说明 |
| --- | --- |
| `invalid_input` | 输入为空。 |
| `invalid_resume` | HITL 续答无法解析。 |
| `response_not_found` | 续答目标不存在。 |
| `internal_error` | 服务端未预期异常。 |

## Chat Completions

兼容通用 OpenAI Chat Completions 客户端：

```text
POST /v1/chat/completions
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `messages` | array | 是 | Chat 消息数组。 |
| `stream` | boolean | 否 | 默认 `false`。 |
| `model` | string | 否 | 覆盖本次请求模型。 |
| `cwd` | string | 否 | 工具执行目录。 |
| `allow_hitl` | boolean | 否 | 是否允许 HITL。 |
| `aliyun_credentials` | object | 否 | 本次请求使用的 Aliyun AK/SK。 |

示例：

```bash
curl --location "$BASE_URL/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

需要完整工具过程、HITL 和 response 存取时，请使用 `/v1/responses`。

`stream=true` 时，Chat Completions 返回普通 SSE `data: {...}` 行，没有 `event:` 名称；结束时同样返回 `data: [DONE]`。

## Health / Models

```bash
curl "$BASE_URL/health"
curl "$BASE_URL/health/detailed"
curl "$BASE_URL/v1/models"
curl "$BASE_URL/v1/skills"
```

修改全局默认模型：

```bash
curl -X POST "$BASE_URL/v1/models/active" \
  --header 'Content-Type: application/json' \
  --data '{"model":"qwen-plus"}'
```

`/v1/responses` 和 `/v1/chat/completions` 里的 `model` 只影响当次请求，不修改全局默认值。
