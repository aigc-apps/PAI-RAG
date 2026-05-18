# MiniAgent

MiniAgent 是一个 OpenAI compatible 的自我进化 Agent 项目，包含 FastAPI 后端、Next.js Web 前端和 ACP/JSON-RPC 入口。日常使用推荐启动 **FastAPI 后端 + Next.js 前端**。

技术设计、模块说明和扩展细节见 [TECHNICAL.md](./TECHNICAL.md)。

## 目录

- [环境要求](#环境要求)
- [配置](#配置)
- [启动 Docker 镜像](#启动-docker-镜像)
- [启动 Web 版本](#启动-web-版本)
- [AgentArena 子服务](#agentarena-子服务)
- [其他入口](#其他入口)
- [API 测试](#api-测试)
- [常见问题](#常见问题)
- [提交前检查](#提交前检查)

## 环境要求

- Python 3.11+
- Node.js 20+
- npm
- Redis（仅 `RUNNER_BACKEND=celery` 多 worker 模式需要）

安装 Python 依赖：

```bash
cd PAI-RAG
pip install -r requirements.txt
```

安装前端依赖：

```bash
cd frontends/react
npm install
```

## 配置

复制环境变量模板：

```bash
cd PAI-RAG
cp .env.example .env
```

编辑 `.env`，至少填写：

```env
API_KEY=sk-...
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen-plus
WORKSPACE_ROOT=./workspaces
```

主服务默认是单用户免登录模式，会按 `session_id` 区分会话。工具执行默认使用项目根目录或请求里的 `cwd`；如需强制限制到 `WORKSPACE_ROOT/<user_id>/<session_id>/` 可设置：

```env
ENFORCE_WORKSPACE_FOR_SERVER=true
MAX_GLOBAL_RUNS=20
MAX_USER_RUNS=2
```

长期 memory 默认写入服务用户目录。若要调整用户 memory 行为，可以配置：

```env
ENABLE_LONG_TERM_MEMORY_FOR_USERS=true
ENABLE_SHARED_MEMORY_FOR_USERS=false
```


## 启动 Docker 镜像

镜像不会内置 `.env`、本地凭证或 `skills/`，运行时用 `--env-file` 和 volume 注入：

```bash
docker build -t pai-rag:local .
docker run --rm --env-file .env -p 8680:8680 -v "$PWD/skills:/app/skills:ro" pai-rag:local
```

浏览器访问：

```text
http://localhost:8680
```

## 启动 Web 版本

默认 `RUNNER_BACKEND=thread`，适合本地开发。

### 1. 启动 FastAPI 后端

在项目根目录运行：

```bash
cd PAI-RAG
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

后端健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果要支持多 Uvicorn worker，需要切到 Celery 模式，并先启动 Redis 与 worker：

```bash
redis-server
```

```bash
cd PAI-RAG
RUNNER_BACKEND=celery REDIS_URL=redis://127.0.0.1:6379/0 \
celery -A backend.celery_app worker --loglevel=info --concurrency=4
```

```bash
cd PAI-RAG
RUNNER_BACKEND=celery REDIS_URL=redis://127.0.0.1:6379/0 \
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. 启动 Next.js 前端

另开一个终端：

```bash
cd PAI-RAG/frontends/react
BACKEND_BASE_URL=http://127.0.0.1:8000 npm run dev
```

本机浏览器访问：

```text
http://127.0.0.1:3000
```

远程服务器开发访问时，让 Next.js 监听所有网卡：

```bash
cd PAI-RAG/frontends/react
BACKEND_BASE_URL=http://127.0.0.1:8000 npm run dev -- --hostname 0.0.0.0 --port 3001
```

浏览器访问：

```text
http://your-server-ip:3001
```

首次打开 Web 前端后可直接开始对话。聊天记录按 `session_id` 区分保存。

## AgentArena 子服务

AgentArena 是独立的双 Agent 对比页面，源码在 `services/agent-arena/`。它有自己的 `.env`，不要和主项目 `.env` 混用。

初始化配置：

```bash
cd PAI-RAG
cp services/agent-arena/.env.example services/agent-arena/.env
```

编辑 `services/agent-arena/.env`，至少配置 `ARENA_API_KEY`、两个 Agent 的 `AGENT_*_BASE_URL` / `AGENT_*_MODEL` / `AGENT_*_API_KEY`，以及可选的 Judge 配置。

AgentArena 不由主项目 `scripts/start.sh` 管理，单独启动。常用方式是两个终端分别拉起：

```bash
# 终端 1：启动 PAI-RAG
./scripts/start.sh --dev --port 3001 --backend-port 8000
```

```bash
# 终端 2：启动 AgentArena
cd services/agent-arena
./start.sh --host 0.0.0.0 --port 8787
```

访问 AgentArena：

```text
http://127.0.0.1:8787
```

## API 测试

后端默认免登录。会话由前端在调用 `/v1/responses` 时携带 `session_id` 维持，无需单独的 run 资源：

```bash
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8000/v1/sessions \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl --no-buffer --location 'http://127.0.0.1:8000/v1/responses' \
  --header 'Content-Type: application/json' \
  --data "{\"session_id\":\"${SESSION_ID}\",\"input\":\"你好，请用一句话介绍你自己\",\"stream\":true}"
```

两个 OpenAI 兼容入口当前都只支持流式调用；传 `stream:false` 会返回 `400 unsupported_mode`。

流式 Chat Completions：

```bash
curl --no-buffer --location 'http://127.0.0.1:8000/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'X-Session-Id: test-session-001' \
  --data '{
    "model": "pairag-agent",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    "stream": true
  }'
```

`stream:true` 时只返回 OpenAI Chat Completions 兼容的 `delta.content` SSE；不会输出工具调用和思考步骤。需要看到工具调用、工具结果请改用 `/v1/responses`。

流式 Responses API 会输出 `response.created`、`response.output_text.delta`、`response.output_item.*`、`response.completed` 等事件：

```bash
curl --no-buffer --location 'http://127.0.0.1:8000/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "input": "检查当前 workspace 下有哪些文件",
    "stream": true
  }'
```

普通多轮推荐记录上一轮 `response.completed.id`，下一轮作为 `previous_response_id` 传回：

```bash
curl --no-buffer --location 'http://127.0.0.1:8000/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "previous_response_id": "resp_xxx",
    "input": "继续上一轮，补充风险点",
    "stream": true
  }'
```

默认新请求是自主模式：`allow_hitl:false`，Agent 不会暂停等待用户。需要前端弹出确认/补充输入时显式传 `allow_hitl:true`；暂停后用 `previous_response_id + function_call_output` 续答：

```bash
curl --no-buffer --location 'http://127.0.0.1:8000/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "pairag-agent",
    "input": "如缺少关键信息，请暂停询问我",
    "allow_hitl": true,
    "stream": true
  }'
```

```bash
curl --no-buffer --location 'http://127.0.0.1:8000/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "previous_response_id": "resp_pause",
    "input": [{"type":"function_call_output","call_id":"call_ask_001","output":"继续执行"}],
    "stream": true
  }'
```

Session API：

```bash
curl http://127.0.0.1:8000/v1/sessions
curl -X POST http://127.0.0.1:8000/v1/sessions
curl http://127.0.0.1:8000/v1/sessions/test-session-001
curl -X DELETE http://127.0.0.1:8000/v1/sessions/test-session-001
```
