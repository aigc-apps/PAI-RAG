# PAI-RAG Agent API 调用指南（以"校验引擎配置"为例）

以PAI-REC AI诊断运维任务为例，说明 PAI-RAG Agent 服务 HTTP API 的完整调用流程。

> **示例任务**：校验PAI-REC引擎配置 `embedding_config`，instanceId `pairec-cn-inner-khhjd7wnn1geomcirl`，region `cn-beijing`，环境 `生产`，status `Released`。

```bash
export BASE_URL="<your-eas-endpoint>"
```

## 0. 接口选型

| 接口 | 适用场景 |
| --- | --- |
| `/v1/runs` | 完整 Agent 生命周期事件流（推理 → 工具调用 → 工具结果 → 后续推理）；支持 `ask_user` 中断 |
| `/v1/chat/completions` | OpenAI 兼容入口；只需要文本回答，不暴露中间过程 |
| `/v1/responses` | 返回完整 `output[]`（含工具调用条目）；不需要中途状态 |

下文以 `/v1/runs` 为主，备用接口见 §3。

---

## 1. /v1/runs

### 1.1 session_id 获取

`session_id` 是多轮对话和 ask_user 续接的唯一标识，所有调用均需保留。两种来源：

| 策略 | 实现 | 取值位置 |
| --- | --- | --- |
| 服务端分配（推荐） | POST 时不传 `session_id`，从响应中读取 | `stream=false` 时为响应 JSON 的 `session_id`；`stream=true` 时为响应 header `X-Session-Id` |
| 客户端生成 | 客户端生成 UUID 并在 POST body 的 `session_id` 字段传入；服务端首次见到时 lazy-create | 客户端本地 |

格式：`^[A-Za-z0-9_-]+$`，UUID 满足。被其他 user 占用时返回 404。

服务端分配（`stream=false`，从响应 JSON 读取）：

```bash
RESP=$(curl -s -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "请校验引擎配置 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region cn-beijing。校验通过后告诉我有没有偏离基线。"
  }')
SESSION_ID=$(echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
RUN_ID=$(echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
```

服务端分配（`stream=true`，从响应 header 读取）：

```bash
curl -s -D /tmp/runs.h --no-buffer -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{"input":"校验 embedding_config","stream":true}' > /tmp/runs.sse
SESSION_ID=$(awk 'BEGIN{IGNORECASE=1}/^x-session-id:/{print $2}' /tmp/runs.h | tr -d '\r')
RUN_ID=$(awk 'BEGIN{IGNORECASE=1}/^x-run-id:/{print $2}' /tmp/runs.h | tr -d '\r')
```

客户端生成：

```bash
SESSION_ID=$(python3 -c 'import uuid;print(uuid.uuid4())')

curl -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"校验 embedding_config\"
  }"
```

事件订阅有两种模式：

- `stream=true`：响应 body 直接为 SSE。
- `stream=false`（默认）：POST 立即返回 JSON（含 `run_id`），随后通过 `GET /v1/runs/{run_id}/events` 订阅 SSE，支持断点续传。

### 1.2 多轮对话

多轮通过复用同一个 `session_id` 实现，每轮独立 POST `/v1/runs`。服务端维护该 session 的完整历史，新 input 无需重复携带 instanceId / region 等上下文。第一轮由服务端分配 `session_id`，后续轮次客户端把它带回去：

```bash
# 第一轮：不带 session_id，从响应读取
RESP=$(curl -s -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{"input":"校验 embedding_config"}')
SESSION_ID=$(echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# 第二轮：复用上一轮的 session_id（待第一轮 run.completed 后发起）
curl -s -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"再校验同实例的 ranker_config，列出差异\"}"
# → 新 run_id；session_id 不变
```

客户端需维护两项状态：

- `session_id`：会话级，整个 session 内不变
- `run_id`：每轮一个，从 POST 响应中获取最新值

约束：上一轮 `status ∈ {running, started}` 时再 POST 同一 `session_id` 返回 `409 session_busy`；并发请求需新建 session。

### 1.3 本轮覆盖模型

POST `/v1/runs` 支持 `model` 字段，仅对当次 run 覆盖上游 LLM。该字段不写入会话默认值，下一轮不传则回退至全局生效模型（`/v1/models/active` 的值）。

适用场景：单次复杂校验或排障临时使用更强模型（如 `qwen-max`），不污染该 session 后续轮次和其他 session。

```bash
curl -X POST "$BASE_URL/v1/runs" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"input\": \"校验 embedding_config 是否偏离基线\",
    \"model\": \"qwen-max\"
  }"
```

`model` 校验规则：非空字符串、不含空白、长度 ≤ 200。服务端不预校验上游是否支持该模型，由上游返错透传。具体路由到哪个 provider 由 `memory/runtime.json` 的前缀规则决定，详见 §2 与 `API.md` 的"多 Provider / Key 池"章节。

`/v1/chat/completions` 与 `/v1/responses` 的 `model` 字段语义一致，均为本次请求覆盖、不写入全局。

### 1.4 ask_user：Agent 中途反问的续接

Agent 执行中如需用户补充输入，会发出 `ask_user` 事件并暂停，run 进入 `waiting_user` 状态。本次 SSE 流随后以 `done(end_turn)` 结束，不会出现 `run.completed` 与 `[DONE]` 之前的最终答案。

```text
data: {"event":"ask_user","run_id":"run_xxx",
       "question":"embedding_config 同时存在 Released（v3）和 Staging（v4），校验哪一个？",
       "candidates":["Released（v3）","Staging（v4）","两个都校验"]}
```

回答方式与发起新一轮一致：使用同一 `session_id` 再次 POST `/v1/runs`，`input` 即为答案。服务端检测到 `waiting_user` 状态时，将该 input 注入正在等待的 run，**`run_id` 不变**：

```bash
curl -X POST "$BASE_URL/v1/runs" \
  -d "{\"session_id\":\"${SESSION_ID}\",\"input\":\"Released（v3）\"}"
```

随后客户端需重新订阅同一 `run_id` 的 events 流（原 SSE 连接已关闭），Agent 从 `waiting_user` 恢复执行，直至下一次 `ask_user` 或 `run.completed`。

```
                       ┌── 上一轮已完成 → 新 run_id
                       │
POST /v1/runs ─────────┤
(同 session_id)        │
                       └── waiting_user 时 → 注入回答，run_id 不变
```

新一轮与续答两种语义由服务端按当前 session `status` 自动区分，客户端代码路径完全一致。

### 1.5 SSE 事件类型

所有事件携带 `run_id` 与 `timestamp`，多数事件还携带 `step_id`，用于将工具调用挂到对应推理步骤。

| `event` | 关键字段 |
| --- | --- |
| `reasoning.started` / `reasoning.available` / `reasoning.completed` | `step_id`、`text`、`status` |
| `tool.delta` | `tool_call_id`、`tool`、`arguments_delta`、`arguments_text` |
| `tool.started` / `tool.updated` / `tool.completed` | `tool_call_id`、`tool`、`status`、`input` / `content` |
| `message.delta` | `delta`（最终回答增量） |
| `ask_user` | `question`、`candidates` |
| `run.completed` | `output`、`usage` |
| `run.failed` | `error` |

最终答案有两种获取方式：累加 `message.delta` 的 `delta` 字段，或直接读取 `run.completed.output`。

事件流片段：

```text
data: {"event":"reasoning.available","step_id":"model-1","text":"先调 DescribeEngineConfig 拉当前配置..."}
data: {"event":"tool.started","tool_call_id":"call_1_0","tool":"exec_command","input":{"cmd":"aliyun pai DescribeEngineConfig --InstanceId pairec-cn-inner-khhjd7wnn1geomcirl --ConfigName embedding_config"}}
data: {"event":"tool.completed","tool_call_id":"call_1_0","status":"completed","content":"{\"ConfigName\":\"embedding_config\",\"Status\":\"Released\",\"Version\":\"v3\"}"}
data: {"event":"message.delta","delta":"已完成校验，未发现偏离..."}
data: {"event":"run.completed","output":"...","usage":{...}}
data: [DONE]
```

### 1.6 状态查询、停止、断点续传

```bash
curl "$BASE_URL/v1/runs/${RUN_ID}"                                                  # 查询状态
curl -X POST "$BASE_URL/v1/runs/${RUN_ID}/stop"                                     # 主动终止
curl --no-buffer "$BASE_URL/v1/runs/${RUN_ID}/events?last_event_id=1778640000123-0" # 断点续传
```

`status` 取值：`started` / `running` / `waiting_user` / `completed` / `failed` / `cancelled`。

**断点续传机制**

- `last_event_id` 形如 `<13 位毫秒时间戳>-<序号>`，celery 模式由 Redis Stream 生成、thread 模式由服务端合成同样格式（客户端不区分后端实现）。
- SSE 流的每条 `data:` 事件前都带一行 `id: <cursor>`。使用浏览器 `EventSource` 时 `lastEventId` 会自动在重连请求里以 `Last-Event-ID:` 头回填，无需客户端手工管理 cursor。
- 服务端为每个 run 维护一份"已下发到哪"的 cursor。客户端 GET `/v1/runs/{run_id}/events` 不传 `last_event_id` 时用该 cursor 续读；显式传入 `?last_event_id=...` 查询参数或 `Last-Event-ID:` 头会覆盖该 cursor。
- 也可通过 `GET /v1/runs/{run_id}` 读响应里的 `last_event_id` 字段拿到当前最新 cursor，作为首次连接时的指定起点。

### 1.7 客户端伪代码

```python
import json, uuid, requests

def stream_run(run_id):
    """订阅 run_id 的 SSE，处理事件直至 ask_user 或终态。

    返回值：
      - ask_user 事件本身：表示需要用户补答案
      - None：表示 run 终态（completed / failed / cancelled）
    """
    with requests.get(f"{BASE_URL}/v1/runs/{run_id}/events", stream=True) as r:
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                return None
            evt = json.loads(payload)
            if evt.get("event") == "ask_user":
                return evt
            if evt.get("event") in ("run.completed", "run.failed"):
                return None
            render(evt)
    return None

def chat_one_turn(session_id, user_input, model=None):
    body = {"session_id": session_id, "input": user_input}
    if model:
        body["model"] = model
    r = requests.post(f"{BASE_URL}/v1/runs", json=body)
    run_id = r.json()["run_id"]
    while True:
        ask = stream_run(run_id)
        if ask is None:
            return                                       # run 终态
        answer = prompt_user(ask["question"], ask.get("candidates", []))
        requests.post(f"{BASE_URL}/v1/runs",
                      json={"session_id": session_id, "input": answer})
        # run_id 不变；服务端使用其 cursor 自动续读，无需显式传 last_event_id

session_id = str(uuid.uuid4())
chat_one_turn(session_id, "校验 embedding_config，输出对照基线的差异", model="qwen-max")
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

- 立即对所有新建 run / response / chat 生效。
- 写入 `memory/runtime.json` 的 `active_model` 字段持久化（与多 provider / key 池配置共用同一文件）。
- HTTP server / Celery worker / ACP server 跨进程共享（基于文件 mtime 失效重读）。
- 不打断已经在跑的 stream。

约束：模型名 ≤ 200 字符、不含空白；服务端不预校验上游是否支持该模型。

回退默认值：直接 POST 当前环境变量 `MODEL` 的值（默认 `qwen-plus`），无独立 reset 接口。

### 2.2 优先级链

```
请求 body.model（per-run / per-request）   ── 仅本次请求生效
       ↓ 未传
全局 active_model（/v1/models/active）       ── 跨进程持久化
       ↓ 未设置
环境变量 MODEL                                ── 启动时默认
       ↓ 未设置
qwen-plus                                    ── 内置兜底
```

请求体 `model` 字段不写入全局，下一次不传即沿此链回退。`/v1/runs`、`/v1/chat/completions`、`/v1/responses` 三处的 body.model 语义一致。

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

## 3. 备用接口

### 3.1 `/v1/chat/completions`（OpenAI 兼容）

仅输出最终文本，不暴露工具调用与思考步骤。多轮通过 `X-Session-Id` 复用 session。`model` 字段语义同 `/v1/runs`，仅本次请求覆盖。

```bash
curl "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "X-Session-Id: ${SESSION_ID}" \
  -d '{"model":"qwen-max","messages":[{"role":"user","content":"校验 embedding_config"}]}'
```

### 3.2 `/v1/responses`（结构化结果）

`output[]` 包含 `function_call` / `function_call_output` / `message` 三类条目。**不支持 `ask_user`**；需要中断式交互请改用 `/v1/runs`。多轮可通过 `previous_response_id` / `conversation` / `session_id` 三选一。`model` 字段仅本次请求覆盖。
