# Backend API

This is the backend API for the application. It is built on fastapi.

## Get started

1. Prerequisites

Before you begin, make sure you have the following requirements:

- Python (3.11-3.12) with poetry installed.
- Redis server running in local or a remote server.

2. Install dependencies

```bash
poetry install
```

3. Start locally

```bash
./scripts/start.sh --frontend-port 8680 --backend-port 8682 --dev
```

Open http://localhost:8680 in your browser.

## 数据库schema更新

我们使用`alembic`工具管理数据库的schema更新。当有更新时，请执行

```sh
alembic revision --autogenerate -m "YOUR CHANGE MESSAGE"
```

然后按上述命令启动服务即可。
