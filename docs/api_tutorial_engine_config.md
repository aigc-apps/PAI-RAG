# PAI-RAG Agent API 调用指南（以"校验引擎配置"为例）

以PAI-REC AI诊断运维任务为例，说明 PAI-RAG Agent 服务 HTTP API 的完整调用流程。

> **示例任务**：校验PAI-REC引擎配置 `embedding_config`，instanceId `pairec-cn-inner-khhjd7wnn1geomcirl`，region `cn-beijing`，环境 `生产`，status `Released`。

```bash
export BASE_URL="<your-eas-endpoint>"
```

## 0. 接口选型

| 接口 | 适用场景 |
| --- | --- |
| `/v1/responses` | OpenAI Responses 兼容；流式回看完整 Agent 生命周期事件（推理 → 工具调用 → 工具结果 → 最终回答），显式 `allow_hitl=true` 后支持 HITL 中断与续答 |
| `/v1/chat/completions` | OpenAI 兼容入口；只需要文本回答，不暴露中间过程 |

下文以 `/v1/responses` 为主，`/v1/chat/completions` 见 §3。

> 历史接口 `/v1/runs` 已在 SDK 迁移过程中整体删除，请改用 `/v1/responses`。
>
> 默认新请求是自主模式：`allow_hitl=false`，Agent 会自行推进任务。只有显式传 `allow_hitl:true` 时，才允许 `ask_user` / 人工审批工具把流暂停为 `requires_action`。

---

## 1. /v1/responses

### 1.1 推荐的单轮调用

后端集成优先使用 `/v1/responses`，并把业务参数一次性写完整。不要只传“校验 embedding_config”，否则 Agent 可能需要额外推断环境、状态或实例。

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "pairec-embedding-config-check-001",
    "input": "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。请列出匹配配置版本，获取 Released 配置并运行配置校验；最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。",
    "stream": true
  }'
```

> `/v1/responses` 仅支持流式（`stream=true`）。如果配置本身存在校验错误，本次 API 调用仍会以 `response.completed` 正常结束，最终文本会说明“配置校验失败”和错误摘要；不要把 `response.completed` 等同于业务校验通过。

### 1.2 session_id 获取（Legacy / 前端兼容）

`session_id` 可用于前端或老客户端复用历史；外部业务集成更推荐使用 `conversation` 或上一轮 `previous_response_id`。两种来源：

| 策略 | 实现 | 取值位置 |
| --- | --- | --- |
| 服务端分配 | 先 POST `/v1/sessions` 拿到 `session_id`，每次请求带回去 | 响应 JSON 的 `session_id` |
| 客户端生成 | 客户端生成 UUID 当作 `session_id`；服务端首次见到时 lazy-create | 客户端本地 |

格式：`^[A-Za-z0-9_-]+$`，UUID 满足。被其他 user 占用时返回 404。

服务端分配：

```bash
SESSION_ID=$(curl -s -X POST "$BASE_URL/v1/sessions" -d '{}' \
  -H 'Content-Type: application/json' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。\",
    \"stream\": true
  }"
```

客户端生成：

```bash
SESSION_ID=$(python3 -c 'import uuid;print(uuid.uuid4())')

curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。\",\"stream\":true}"
```

### 1.3 多轮对话

多轮可以复用同一个 `session_id`，每轮独立 POST `/v1/responses`。服务端维护该 session 的完整历史，新 input 无需重复携带 instanceId / region 等上下文。后端集成也可以使用上一轮 `response.completed.id` 作为 `previous_response_id` 并继续传普通文本；这会启动一个新的 response。只有 `previous_response_id` 搭配 `function_call_output` / `mcp_approval_response` 输入项时，才表示 HITL resume。

```bash
# 第一轮
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。\",\"stream\":true}"

# 第二轮：复用上一轮的 session_id（待第一轮 response.completed 后发起）
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"再校验同实例的 ranker_config，列出差异\",\"stream\":true}"
```

约束：上一轮仍在运行中时再 POST 同一 `session_id` 返回 `409 session_busy`；并发请求需新建 session。HITL 暂停属于 run 状态，不再占用 session 的 running 状态，续答必须走 `previous_response_id + function_call_output`。

### 1.4 本轮覆盖模型

POST `/v1/responses` 支持 `model` 字段，仅对当次 response 覆盖上游 LLM。该字段不写入会话默认值，下一轮不传则回退至全局生效模型（`/v1/models/active` 的值）。

适用场景：单次复杂校验或排障临时使用更强模型（如 `qwen-max`），不污染该 session 后续轮次和其他 session。

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。\",
    \"model\": \"qwen-max\",
    \"stream\": true
  }"
```

`model` 校验规则：非空字符串、不含空白、长度 ≤ 200。服务端不预校验上游是否支持该模型，由上游返错透传。具体路由到哪个 provider 由 `memory/runtime.json` 的前缀规则决定，详见 §2 与 `API.md` 的"多 Provider / Key 池"章节。

`/v1/chat/completions` 与 `/v1/responses` 的 `model` 字段语义一致，均为本次请求覆盖、不写入全局。

### 1.5 ask_user：Agent 中途反问的续接

默认自主模式下，Agent 不会因为 `ask_user` 停下来等待用户；它会按工具参数里的 `default_action` 或安全兜底指令继续。需要前端弹出用户选择/补充输入时，启动请求必须传 `allow_hitl:true`：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。如果同时存在多个可校验版本，请暂停让我选择；最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。\",
    \"allow_hitl\": true,
    \"stream\": true
  }"
```

此时 Agent 执行中如需用户补充输入（如 `ask_user` 工具），会以 `response.requires_action` + `response.incomplete` 终结当前流，附带 `submit_tool_outputs.tool_calls[]`。客户端续答时携带 `previous_response_id` + `function_call_output` 输入项即可：

```text
event: response.requires_action
data: {"id":"resp_xxx","status":"requires_action",
       "required_action":{"type":"submit_tool_outputs",
         "submit_tool_outputs":{"tool_calls":[{
           "id":"call_abc","type":"function",
           "function":{"name":"ask_user","arguments":"{\"question\":\"embedding_config 同时存在 Released 和 Staging，校验哪一个？\",\"candidates\":[\"Released\",\"Staging\",\"两个都校验\"]}"}
         }]}}}
event: response.incomplete
data: {"id":"resp_xxx","status":"incomplete",
       "incomplete_details":{"reason":"requires_action"},
       "required_action":{...同上...}}
data: [DONE]
```

恢复执行：把 `response.requires_action` 中的 `id` 作为 `previous_response_id`，把回答包成 `function_call_output` 输入项再次 POST：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{
    \"previous_response_id\": \"resp_xxx\",
    \"input\": [
      {\"type\":\"function_call_output\",\"call_id\":\"call_abc\",\"output\":\"Released\"}
    ],
    \"stream\": true
  }"
```

服务端从暂停点恢复执行，直至下一次 `requires_action` 或 `response.completed`。同一个暂停 response 只能被成功续答一次；重复续答或对非暂停 response 续答会返回 `409 not_resumable`。

### 1.6 SSE 事件类型

OpenAI Responses 兼容 SSE，每帧形如：

```text
event: <type>
data: <json>
```

| `type` | 关键字段 |
| --- | --- |
| `response.created` | `id`、`status='in_progress'` |
| `response.output_item.added` | `output_index`、`item.{type, id}` — `type` ∈ {`message`, `function_call`} |
| `response.output_text.delta` | `delta`（最终回答增量） |
| `response.function_call_arguments.delta` | `item_id`、`delta`（工具参数流式） |
| `response.function_call_arguments.done` | `item_id`、`arguments`（工具参数完成） |
| `response.output_item.done` | `item.{type, id, ...}` |
| `response.reasoning_step.started` / `completed` | `step_id`（合成的思考步骤边界，前缀 `rs_synth_`） |
| `response.requires_action` | `allow_hitl=true` 且暂停时出现；`id`、`required_action.submit_tool_outputs.tool_calls[]` |
| `response.completed` | `output[]`、`usage` |
| `response.failed` | `error` |

最终答案有两种获取方式：累加 `response.output_text.delta` 的 `delta` 字段，或直接读取 `response.completed.output[]` 中 `type='message'` 的 `content[].text`。

事件流片段：

```text
event: response.created
data: {"type":"response.created","id":"resp_xxx","status":"in_progress","model":"qwen-plus"}
event: response.reasoning_step.started
data: {"type":"response.reasoning_step.started","step_id":"rs_synth_resp_xxx_1","synthetic":true}
event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"exec_command","arguments":""}}
event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"cmd\":\"aliyun pairecservice list-engine-configs"}
event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"cmd\":\"aliyun pairecservice list-engine-configs --instance-id pairec-cn-inner-khhjd7wnn1geomcirl --environment Prod --status Released --name embedding_config --region cn-beijing\"}"}
event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fco_1","type":"function_call_output","call_id":"call_1","output":"{\"EngineConfigs\":[{\"Name\":\"embedding_config\",\"Environment\":\"Prod\",\"Status\":\"Released\",\"EngineConfigId\":\"487\",\"Version\":\"20260514095521\"}],\"TotalCount\":1}"}}
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"配置校验完成：未通过，发现若干错误..."}
event: response.completed
data: {"type":"response.completed","id":"resp_xxx","status":"completed","output":[...],"usage":{...}}
data: [DONE]
```

示例中的 `response.completed` 只表示 Agent 运行完成，不表示配置业务校验通过。最终业务结论以 `response.output_text.delta` 累计文本或 `response.completed.output[]` 中的最终消息为准。

### 1.7 状态查询、停止、刷新恢复

```bash
curl "$BASE_URL/v1/responses/${RESPONSE_ID}"               # 查询某次 response
curl "$BASE_URL/v1/sessions/${SESSION_ID}"                 # 查询会话消息和状态
curl -X POST "$BASE_URL/v1/sessions/${SESSION_ID}/cancel"  # 主动终止当前 in-flight response
```

- 客户端断线重连：普通完成/失败结果可从 `GET /v1/sessions/{session_id}` 的 `messages` 恢复；HITL 暂停需保存 `response.requires_action.id`，再用 `GET /v1/responses/{response_id}` 恢复 `required_action`。
- 取消：`POST /v1/sessions/{session_id}/cancel` 会让正在跑的 SSE 流以 `response.failed` 终结。
- `/v1/responses` 不支持断点续传 cursor —— 同 session 重发会得到一条新 response。

### 1.8 客户端伪代码

```python
import json, uuid, requests

BASE_URL = "<your-eas-endpoint>"

def stream_response(body):
    """订阅 /v1/responses SSE，遇到 requires_action 就返回 pending；
    遇到 completed/failed 就返回 None。"""
    pending = None
    with requests.post(f"{BASE_URL}/v1/responses",
                       json=body, stream=True,
                       headers={"Accept": "text/event-stream"}) as r:
        event_name = None
        for raw in r.iter_lines():
            if not raw:
                event_name = None
                continue
            if raw.startswith(b"event: "):
                event_name = raw[7:].decode()
            elif raw.startswith(b"data: "):
                payload = raw[6:]
                if payload == b"[DONE]":
                    return pending
                evt = json.loads(payload)
                if event_name == "response.requires_action":
                    fc = evt["required_action"]["submit_tool_outputs"]["tool_calls"][0]
                    pending = {
                        "response_id": evt["id"],
                        "call_id": fc["id"],
                        "tool_name": fc["function"]["name"],
                        "arguments": json.loads(fc["function"]["arguments"] or "{}"),
                    }
                elif event_name == "response.output_text.delta":
                    print(evt["delta"], end="", flush=True)
    return pending

def chat_one_turn(session_id, user_input, model=None, allow_hitl=False):
    body = {"session_id": session_id, "input": user_input, "stream": True}
    if allow_hitl:
        body["allow_hitl"] = True
    if model:
        body["model"] = model
    pending = stream_response(body)
    while pending is not None:
        answer = prompt_user(pending["arguments"].get("question", ""),
                             pending["arguments"].get("candidates", []))
        pending = stream_response({
            "previous_response_id": pending["response_id"],
            "input": [{"type": "function_call_output",
                       "call_id": pending["call_id"], "output": answer}],
            "stream": True,
        })

session_id = str(uuid.uuid4())
chat_one_turn(session_id, "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。", model="qwen-max", allow_hitl=True)
chat_one_turn(session_id, "再校验 ranker_config")
```

---

## 2. 模型管理

### 2.1 全局生效模型

服务端维护一个全局生效模型，所有新建对话默认使用。运行中的 stream 沿用旧模型直至结束。

```bash
curl "$BASE_URL/v1/models"                                       # 查询
curl -X POST "$BASE_URL/v1/models/active" \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen-max"}'                                     # 切换
```

效果：

- 立即对所有新建 response / chat 生效。
- 写入 `memory/runtime.json` 的 `active_model` 字段持久化（与多 provider / key 池配置共用同一文件）。
- HTTP server / Celery worker / ACP server 跨进程共享（基于文件 mtime 失效重读）。
- 不打断已经在跑的 stream。

约束：模型名 ≤ 200 字符、不含空白；服务端不预校验上游是否支持该模型。

回退默认值：直接 POST 当前环境变量 `MODEL` 的值（默认 `qwen-plus`），无独立 reset 接口。

### 2.2 优先级链

```
请求 body.model（per-request）              ── 仅本次请求生效
       ↓ 未传
全局 active_model（/v1/models/active）       ── 跨进程持久化
       ↓ 未设置
环境变量 MODEL                                ── 启动时默认
       ↓ 未设置
qwen-plus                                    ── 内置兜底
```

请求体 `model` 字段不写入全局，下一次不传即沿此链回退。`/v1/responses` 与 `/v1/chat/completions` 两处的 body.model 语义一致。

### 2.3 多 Provider / Key 池

通过 `memory/runtime.json` 的 `providers` 段可配置多个上游 provider 与每 provider 下的 key 池：

- 按 model 名前缀路由到 provider（例：`qwen-*` → 阿里云 DashScope，`glm-*` → 智谱）。
- provider 下的 key 池按 round-robin 分配。
- key 健康度按 HTTP 状态码维护：401/403 永久 evict、429 进入 300 秒冷却（到期自动复活）、5xx 透传不剔。
- LLMClient 在可恢复错误（401/403/429）下自动从池中取下一把 key 重试，最多 2 次轮换。
- 文件 mtime 变化即热加载，无需重启进程。
- 不存在该文件时回退到环境变量 `API_KEY` / `API_BASE` 拼一个默认 `qwen` provider。

完整 schema、路由规则与失败语义见 `API.md` 的"多 Provider / Key 池"章节，本文不再重复。

---

## 3. 备用接口：`/v1/chat/completions`（OpenAI 兼容）

仅输出最终文本，不暴露工具调用与思考步骤。多轮通过 `X-Session-Id` 复用 session。`model` 与 `allow_hitl` 字段语义同 `/v1/responses`，仅 `model` 是本次请求覆盖。

```bash
curl --no-buffer "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "X-Session-Id: ${SESSION_ID}" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。"}],"stream":true}'
```

中断式交互（ask_user）需要显式 `allow_hitl:true`，通过 chat 兼容线的 `__ask_user__` 协议（`finish_reason='tool_calls'`）+ `role='tool'` 续答消息实现，详见 `API.md`。
