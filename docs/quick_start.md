# [PAI-RAG]安装指南

## 系统要求

- Python 3.11
- Node.js 16+ (推荐使用LTS版本)
- Git
- Conda (推荐Miniconda)

## 方式一：本地开发

1. 克隆项目代码到本地

   ```bash
   git clone https://github.com/aigc-apps/PAI-RAG.git
   cd PAI-RAG
   ```

   > 注意：请确保已安装Git，若未安装请先安装Git再进行此步骤。

2. 创建环境并安装依赖包

   ```bash
   # 创建Python环境
   conda create -n pai-rag-env python=3.11
   conda activate pai-rag-env

   # 安装项目核心依赖
   pip install poetry
   poetry install

   # 安装前端依赖
   cd frontend && npm install
   ```

3. 配置环境变量
   拷贝 `.env.example` 为 `.env`，并修改对应的配置。

   ```bash
   cp .env.example .env
   ```

   编辑.env文件，根据需要修改以下关键配置：

   ```bash
   # 1. Broker 配置，留空则使用本地 Redis，默认值为 redis://localhost:6379/0
   PAIRAG_BROKER=

   # 2. 文件存储配置，填写OSS相关信息
   FILE_STORE_TYPE=oss
   OSS_ACCESS_KEY_ID=
   OSS_ACCESS_KEY_SECRET=
   OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
   OSS_BUCKET=

   # 3. 事务数据库配置，支持本地SQLite3和PostgreSQL
   DB_TYPE=sqlite3 # postgresql, sqlite3
   DB_HOST=
   DB_PORT=5432
   DB_USER=
   DB_PASSWORD=
   DB_NAME=

   # 4. 向量数据库配置，支持本地Chroma，以及阿里云相关产品
   VECTOR_DB_TYPE=local #local, milvus, elasticsearch, postgresql

   # 4.1 使用 Milvus
   MILVUS_HOST=
   MILVUS_PORT=19530
   MILVUS_USER=root
   MILVUS_PASSWORD=
   MILVUS_DATABASE=default


   # 4.2 使用 Elasticsearch
   ELASTICSEARCH_URL=
   ELASTICSEARCH_USER=elastic
   ELASTICSEARCH_PASSWORD=


   # 4.3 使用 使用PostgreSQL
   POSTGRES_HOST=
   POSTGRES_PORT=5432
   POSTGRES_USER=
   POSTGRES_PASSWORD=
   POSTGRES_DATABASE=
   ```

4. 启动服务

   ```bash
   ./scripts/start.sh --frontend-port 3001 --backend-port 8680
   ```

5. 验证安装
   服务启动后：
   前端可通过 http://localhost:3001 访问，显示如下页面则表示安装部署成功，可以尽情使用了。
   ![quick_start](images/quick_start.jpg)

## 常见问题排查

1. 依赖安装问题

- 若poetry install失败，可尝试先运行poetry lock来锁定依赖版本
- 若遇到Node.js相关问题，确保Node.js版本符合要求

2. 环境变量问题

- 确保所有必填环境变量已正确设置，特别是API密钥类配置
- 修改.env后需重启服务使配置生效

3. 端口冲突

- 如遇端口占用，可使用其他端口：

```bash
./scripts/start.sh --frontend-port 3002 --backend-port 8681
```

## 注意事项

- 项目使用Poetry进行依赖管理，不推荐直接使用pip安装依赖
- 如需更新依赖，修改pyproject.toml后运行poetry lock和poetry install
