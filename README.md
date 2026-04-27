# MiniAgent

MiniAgent 是一个 OpenAI compatible 的自我进化 Agent 项目，包含 FastAPI 后端、Next.js Web 前端和 ACP/JSON-RPC 入口。日常使用推荐启动 **FastAPI 后端 + Next.js 前端**。

技术设计、模块说明和扩展细节见 [TECHNICAL.md](./TECHNICAL.md)。

## 目录

- [环境要求](#环境要求)
- [配置](#配置)
- [启动 Web 版本](#启动-web-版本)
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

复制配置模板：

```bash
cd PAI-RAG
cp config_template.py config.py
```

编辑 `config.py`，至少填写：

```python
API_KEY = "sk-..."
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
AUTH_SECRET = "replace-with-a-long-random-string"
WORKSPACE_ROOT = "./workspaces"
```

普通登录用户的工具执行会被限制在 `WORKSPACE_ROOT/<user_id>/<session_id>/` 内。服务级 `SERVER_API_KEY` 调用默认仍可使用请求里的 `cwd`，如需同样隔离可设置：

```python
ENFORCE_WORKSPACE_FOR_SERVER = True
MAX_GLOBAL_RUNS = 20
MAX_USER_RUNS = 2
```

长期 memory 默认只对服务级调用启用。若要给普通登录用户启用长期记忆，会写入用户私有目录 `memory/users/<user_id>/`，不会写入全局 memory：

```python
ENABLE_LONG_TERM_MEMORY_FOR_USERS = True
ENABLE_SHARED_MEMORY_FOR_USERS = False
```


## 启动 Web 版本

默认 `RUNNER_BACKEND = "thread"`，适合本地开发。

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

首次打开 Web 前端后，先注册账号再开始对话。聊天记录会按登录用户隔离保存。

`SERVER_API_KEY` 是服务级 API 密钥，主要用于脚本或 OpenAI compatible 调用。Web 前端的普通用户请求使用登录 token，不需要把 `SERVER_API_KEY` 暴露给浏览器。

```bash
BACKEND_BASE_URL=http://127.0.0.1:8000 \
SERVER_API_KEY=your-server-api-key \
npm run dev -- --hostname 0.0.0.0 --port 3001
```

## 其他入口

ACP/JSON-RPC：

```bash
cd PAI-RAG
python -m frontends.acp
```

Zed 等编辑器可以使用脚本：

```text
/path/to/PAI-RAG/frontends/acp/run.sh
```

## API 测试

先注册或登录获取用户 token：

```bash
curl --location 'http://127.0.0.1:8000/v1/auth/register' \
  --header 'Content-Type: application/json' \
  --data '{"username":"demo","password":"demo1234"}'

TOKEN=$(curl -s --location 'http://127.0.0.1:8000/v1/auth/login' \
  --header 'Content-Type: application/json' \
  --data '{"username":"demo","password":"demo1234"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

Web 前端使用 ACP 风格结构化 SSE：

```bash
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8000/v1/sessions \
  --header "Authorization: Bearer ${TOKEN}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl --no-buffer --location "http://127.0.0.1:8000/v1/agent/sessions/${SESSION_ID}/prompt" \
  --header "Authorization: Bearer ${TOKEN}" \
  --header 'Content-Type: application/json' \
  --data '{"message":"你好，请用一句话介绍你自己"}'
```

非流式 Chat Completions：

```bash
curl --location 'http://127.0.0.1:8000/v1/chat/completions' \
  --header "Authorization: Bearer ${TOKEN}" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "hermes-agent",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ],
    "stream": false
  }'
```

流式 Chat Completions：

```bash
curl --location 'http://127.0.0.1:8000/v1/chat/completions' \
  --header "Authorization: Bearer ${TOKEN}" \
  --header 'Content-Type: application/json' \
  --header 'X-Session-Id: test-session-001' \
  --data '{
    "model": "hermes-agent",
    "messages": [
      {"role": "user", "content": "你能做什么？"}
    ],
    "stream": true
  }'
```

如果使用服务级 `SERVER_API_KEY` 直接调用后端，也可以把上面的用户 token 换成：

```bash
--header 'Authorization: Bearer your-server-api-key'
```

Session API：

```bash
curl --header "Authorization: Bearer ${TOKEN}" http://127.0.0.1:8000/v1/sessions
curl -X POST --header "Authorization: Bearer ${TOKEN}" http://127.0.0.1:8000/v1/sessions
curl --header "Authorization: Bearer ${TOKEN}" http://127.0.0.1:8000/v1/sessions/test-session-001
curl -X DELETE --header "Authorization: Bearer ${TOKEN}" http://127.0.0.1:8000/v1/sessions/test-session-001
```
