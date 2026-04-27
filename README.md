# MiniAgent

MiniAgent 是一个 OpenAI compatible 的 自我进化 Agent 项目，包含 FastAPI 后端、Next.js Web 前端、CLI 入口和 ACP/JSON-RPC 入口。日常使用推荐启动 **FastAPI 后端 + Next.js 前端**。

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
```


## 启动 Web 版本

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

如果后端配置了 `SERVER_API_KEY`，启动前端时也设置同一个值。这个值只在 Next.js 服务端使用，不会暴露给浏览器：

```bash
BACKEND_BASE_URL=http://127.0.0.1:8000 \
SERVER_API_KEY=your-server-api-key \
npm run dev -- --hostname 0.0.0.0 --port 3001
```

## 其他入口

CLI：

```bash
cd PAI-RAG
python main.py
```

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

非流式 Chat Completions：

```bash
curl --location 'http://127.0.0.1:8000/v1/chat/completions' \
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

如果设置了 `SERVER_API_KEY`，加上：

```bash
--header 'Authorization: Bearer your-server-api-key'
```

Session API：

```bash
curl http://127.0.0.1:8000/v1/sessions
curl -X POST http://127.0.0.1:8000/v1/sessions
curl http://127.0.0.1:8000/v1/sessions/test-session-001
curl -X DELETE http://127.0.0.1:8000/v1/sessions/test-session-001
```
