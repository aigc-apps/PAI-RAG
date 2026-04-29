# AgentArena

PAI-RAG 的独立 Agent 竞技场子服务：把同一条输入同时发送给两个 OpenAI-compatible Agent API，并并排展示输出、耗时和错误。

## 功能

- 后端隐藏两个 Agent 的 API key。
- 支持两个 `/v1/chat/completions` endpoint 并发调用。
- 前端基于 React、Tailwind 和 shadcn/ui 风格组件。
- 支持 system prompt、temperature、max tokens。
- 一个 Agent 失败时，另一个 Agent 的结果仍会展示。
- 可选展示 Hermes 风格 `/v1/runs` 中间过程事件。
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
AGENT_A_TRACE_MODE=runs

AGENT_B_NAME=Another Agent
AGENT_B_BASE_URL=http://127.0.0.1:8000
AGENT_B_API_KEY=your-agent-b-key
AGENT_B_MODEL=mini-agent
AGENT_B_TRACE_MODE=chat

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
- `http://127.0.0.1:8642/v1/chat/completions`

如果某个 Agent 不需要 key，对应的 `AGENT_*_API_KEY` 留空即可。

### 中间过程展示

默认：

```bash
AGENT_A_TRACE_MODE=chat
```

这只调用 `/v1/chat/completions`，只能展示最终答案。

如果 Agent 支持 Hermes API Server 的 runs 接口，可以开启：

```bash
AGENT_A_TRACE_MODE=runs
```

开启后后端会调用：

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

### Judge 裁判员

Judge 使用 OpenAI-compatible `/v1/chat/completions`：

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
