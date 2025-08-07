# [PAI-RAG]安装指南

## 方式一：本地开发

1. 克隆项目代码到本地
   ```bash
   git clone https://github.com/aigc-apps/PAI-RAG.git
   ```
2. 创建环境并安装依赖包
   ```bash
   conda create -n pai-rag-env python=3.11
   conda activate pai-rag-env
   pip install poetry
   poetry install
   cd frontend && npm install
   ```
3. 配置环境变量
   拷贝 `.env.example` 为 `.env`，并修改对应的配置。
4. 启动服务
   ```bash
   ./scripts/start.sh --frontend-port 3001 --backend-port 8680
   ```
