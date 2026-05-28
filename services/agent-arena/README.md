# AgentArena

PAI-RAG 的独立 Agent 竞技场子服务：把同一条输入同时发送给两个 Agent API，并并排展示输出、耗时和错误。

## 功能

- 后端隐藏两个 Agent 的 API key。
- 默认使用 `/v1/responses` 并发调用 PAI-RAG Agent。
- 前端基于 React、Tailwind 和 shadcn/ui 风格组件。
- 支持 system prompt、temperature、max tokens。
- 一个 Agent 失败时，另一个 Agent 的结果仍会展示。
- 可选展示外部 Hermes 风格 Agent 的 `/v1/runs` 中间过程事件（仅用于消费外部 Agent；PAI-RAG 主服务已不暴露 `/v1/runs`，对接 PAI-RAG 时请使用 `responses` 模式）。
- 可配置 OpenAI Judge 模型，从答案和公开过程事件两个层面对结果打分。
- 自动保存每次 PK 和每次 Judge 评估到 SQLite，并提供历史对比页面。

## 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
ARENA_API_KEY=change-me-arena-key

AGENT_A_NAME=Hermes Agent
AGENT_A_BASE_URL=http://127.0.0.1:8642
AGENT_A_API_KEY=your-agent-a-key
AGENT_A_MODEL=hermes-agent
AGENT_A_TRACE_MODE=responses

AGENT_B_NAME=Another Agent
AGENT_B_BASE_URL=http://127.0.0.1:8000
AGENT_B_API_KEY=your-agent-b-key
AGENT_B_MODEL=mini-agent
AGENT_B_TRACE_MODE=responses

JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_OPENAI_API_KEY=your-openai-key
JUDGE_MODEL=gpt-4.1-mini

# 可选，默认写入 data/arena_history.sqlite3
HISTORY_DB_PATH=./data/arena_history.sqlite3
```

如果设置了 `ARENA_API_KEY`，所有 `/api/*` 请求都需要携带：

```text
Authorization: Bearer <ARENA_API_KEY>
```

`AGENT_*_BASE_URL` 可以写成以下任一种：

- `http://127.0.0.1:8642`
- `http://127.0.0.1:8642/v1`
- `http://127.0.0.1:8642/v1/responses`

如果某个 Agent 不需要 key，对应的 `AGENT_*_API_KEY` 留空即可。

### 中间过程展示

默认：

```bash
AGENT_A_TRACE_MODE=responses
```

这会调用 `/v1/responses` 并展示 Responses SSE 中的公开过程事件。

如果对接的是外部 OpenAI-compatible Chat Agent，可以显式开启：

```bash
AGENT_A_TRACE_MODE=chat
```

这只调用外部 `/v1/chat/completions`，只能展示最终答案。

如果对接的是外部 Hermes API Server 风格的 Agent（PAI-RAG 主服务不属于此类，已转向 OpenAI Responses 风格），可以开启：

```bash
AGENT_A_TRACE_MODE=runs
```

开启后后端会向该外部 Agent 发起：

```text
POST /v1/runs
GET  /v1/runs/{run_id}/events
```

并展示公开事件：

- `tool.started`
- `tool.completed`
- `reasoning.available`
- `message.delta`
- `run.completed`
- `run.failed`

不会要求模型泄露隐藏 chain-of-thought；只展示 API 明确返回的可公开过程信息。

如果对接的是 OpenClaw Web API，可以开启：

```bash
AGENT_A_TRACE_MODE=openclaw
AGENT_A_BASE_URL=http://openclaw-host:3000
```

OpenClaw 使用 cookie 会话协议，后端会执行 CSRF、登录、创建 session、`POST /api/chat` SSE，并展示 `started`、`status`、`token`、`done` 这些公开过程事件。固定账号可以通过 `OPENCLAW_EMAIL` / `OPENCLAW_PASSWORD`，或对应 agent 的 API key 环境变量写成 `email@example.com:password`。

### Judge 裁判员

Judge 仍使用外部 OpenAI-compatible `/v1/chat/completions`：

```bash
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_OPENAI_API_KEY=your-openai-key
JUDGE_MODEL=gpt-4.1-mini
```

如果你已经设置了 `OPENAI_API_KEY`，也可以不单独设置 `JUDGE_OPENAI_API_KEY`。

Judge 评估最终答案维度：

- 指令遵循
- 正确性
- 完整性
- 可执行性
- 推理质量
- 清晰度
- 安全和风险意识

如果有过程事件，还会评估：

- 工具相关性
- 工具效率
- 证据使用
- 错误恢复
- 过程透明度
- 风险控制

有过程事件时综合分按“最终答案 75% + 过程表现 25%”理解；没有过程事件时只评估最终答案。

### 稳定性测试（Batch）

页面顶部点击「稳定性测试」可对单个 Agent（或两边同时）用同一条 query body 跑多次，看成功率、延迟分布、答案是否稳定。

支持：

- **目标 Agent**：A / B / A+B 并排
- **请求体两种模式**
  - **Form**：复用竞技场的 `input / system / temperature / max_tokens` 四字段
  - **Raw**：粘贴完整 JSON body，直接 POST 到 Agent 的 `/v1/responses`（后端会自动加 `stream: true`）
- **iterations**：1–200；**concurrency**：1–8（`asyncio.Semaphore` 控制）
- **assertion**：substring 或 regex，每次输出独立判定 pass/fail
- **答案指纹聚类**：对每次成功输出做 normalize + sha1[:8]，把相同/相近的答案归到同一簇
- **导出 JSON**：把当前 batch 的 items + summary 下载为本地文件
- **持久化**：每个 batch 写入 SQLite 的 `batch_runs` + `batch_run_items` 两张表，可通过 `GET /api/batch/{batch_id}` 取回

API 示例：

```bash
curl -N http://127.0.0.1:8787/api/batch \
  -H "Authorization: Bearer change-me-arena-key" \
  -H "Content-Type: application/json" \
  -d '{
    "target": "a",
    "mode": "form",
    "iterations": 10,
    "concurrency": 2,
    "form": {
      "input": "请用三句话介绍你自己",
      "system": "你是一个严谨的助手",
      "temperature": 0.2,
      "max_tokens": 800
    },
    "assertion": {"type": "substring", "value": "助手"}
  }'
```

返回 `text/event-stream`，事件类型：`batch.started`、`run.completed`（每个 run 一条）、`batch.completed`（含每个 Agent 的聚合 summary）、`batch.error`。前端断开连接会触发后端取消，已完成的 run 仍会写入 SQLite。

### 历史记录

后端默认把历史记录写入：

```text
data/arena_history.sqlite3
```

可以用 `.env` 覆盖路径：

```bash
HISTORY_DB_PATH=/path/to/arena_history.sqlite3
```

保存内容包括：

- 用户输入、system prompt、temperature、max tokens
- Agent A/B 的完整返回、公开过程事件和 trace summary
- 每次 Judge 的 winner、summary、评分、优劣势和原始 JSON

前端顶部点击“历史记录”即可查看历史列表和详情。

## 启动

AgentArena 是独立服务，不依赖主项目 `start.sh`。先安装 Python 依赖：

```bash
cd services/agent-arena
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

启动单进程服务，脚本会安装缺失的前端依赖、构建前端，并启动 FastAPI：

```bash
./start.sh --port 8787
```

如需远程访问：

```bash
./start.sh --host 0.0.0.0 --port 8787
```

打开：

```text
http://127.0.0.1:8787
```

## 前后端分开开发

后端：

```bash
cd services/agent-arena
source .venv/bin/activate
uvicorn backend.server:app --host 127.0.0.1 --port 8787
```

前端：

```bash
cd services/agent-arena/frontend
npm install
npm run dev
```

打开：

```text
http://127.0.0.1:5173
```

Vite 会把 `/api` 代理到 `http://127.0.0.1:8787`。

## API 测试

```bash
curl -s http://127.0.0.1:8787/api/compare \
  -H "Authorization: Bearer change-me-arena-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "请用三句话介绍你自己",
    "system": "你是一个严谨的助手",
    "temperature": 0.2,
    "max_tokens": 800
  }' | python -m json.tool
```

Judge API 示例：

```bash
curl -s http://127.0.0.1:8787/api/judge \
  -H "Authorization: Bearer change-me-arena-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "请解释长期记忆设计取舍",
    "system": "你是一个严谨的助手",
    "agent_a": {
      "ok": true,
      "name": "Agent A",
      "model": "a",
      "content": "答案 A",
      "latency_ms": 100,
      "error": null,
      "raw_finish_reason": "stop",
      "trace_supported": false,
      "trace_events": [],
      "trace_summary": {"supported": false}
    },
    "agent_b": {
      "ok": true,
      "name": "Agent B",
      "model": "b",
      "content": "答案 B",
      "latency_ms": 120,
      "error": null,
      "raw_finish_reason": "stop",
      "trace_supported": false,
      "trace_events": [],
      "trace_summary": {"supported": false}
    }
  }' | python -m json.tool
```

历史列表：

```bash
curl -s 'http://127.0.0.1:8787/api/history?limit=20' \
  -H "Authorization: Bearer change-me-arena-key" \
  | python -m json.tool
```

历史详情：

```bash
curl -s 'http://127.0.0.1:8787/api/history/<run_id>' \
  -H "Authorization: Bearer change-me-arena-key" \
  | python -m json.tool
```

## 前端错误日志

前端会自动捕获以下错误并上报到后端：

- `window.error`
- `unhandledrejection`
- React ErrorBoundary 捕获的渲染错误
- 页面里主动 catch 到的配置、对比、Judge 请求错误

日志会写入：

```text
logs/frontend.log
```

运行中的 uvicorn 控制台也会打印同样的 JSON 日志。也可以通过接口查看最近日志：

```bash
curl -s 'http://127.0.0.1:8787/api/frontend-logs?limit=80' \
  -H "Authorization: Bearer change-me-arena-key" \
  | python -m json.tool
```

如果页面白屏，先看这个文件：

```bash
tail -n 80 services/agent-arena/logs/frontend.log
```

## 安全说明

这个工具会把请求转发给具备完整工具能力的 Agent。公网或共享环境必须设置强 `ARENA_API_KEY`，并建议只暴露独立的 AgentArena 端口或放到你自己的反向代理后面。
