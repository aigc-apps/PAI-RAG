# Agent Knowledge Base Architecture Design

- **Status**: Draft v0.1
- **Related PRD**: [Agent Knowledge Base Management PRD](./agent_knowledge_base_prd.md)
- **Target stack**: `newbackend` + `newfrontend`
- **Scope**: 知识库存储、API、前端界面、离线导入流程、Agent 检索工具、管理员管理工具。

## 1. 目标

建设一套可被 Agent 安全使用、可被用户管理和验证的知识库系统。

核心能力：

- 支持文件上传、网站、OSS prefix 三类数据源。
- 支持异步导入、解析、切分、embedding、索引。
- 支持文档元数据查询和编辑。
- 支持 chunk 查看、禁用、恢复、重建。
- 支持 Agent 使用多种查询工具：
  - 语义搜索
  - 关键词匹配
  - 标题/目录搜索
  - 日期/元数据过滤
  - 全文 fetch
- 支持管理员通过 API 或管理工具维护知识库。
- 普通用户只能查询被授权知识库，不能修改知识库配置。

## 2. 总体架构

```text
newfrontend
  ├── Knowledge UI
  ├── Agent Settings UI
  └── Chat UI

newbackend
  ├── REST API
  │   ├── Management API       # admin/editor
  │   ├── Query API            # user/agent
  │   └── Job API              # status/cancel/retry
  ├── Knowledge Service
  ├── Ingestion Service
  ├── Retrieval Service
  ├── Agent Tool Registry
  └── Offline Worker

Storage
  ├── SQL DB                   # metadata, jobs, permissions
  ├── Object/File Store         # raw files, normalized text
  ├── Vector Store              # chunk embeddings
  └── Keyword Index             # FTS/BM25/trigram
```

设计原则：

- SQL 是状态源；向量库和关键词索引是可重建投影。
- 原始文件与 normalized text 必须保留，便于重建索引。
- 数据源适配器只负责发现和抓取，下游解析/切分/索引统一处理。
- Agent 只能访问显式绑定且用户有权限的知识库。
- 普通用户查询工具和管理员管理工具必须分离。

## 3. 核心概念

| 概念 | 说明 |
|------|------|
| KnowledgeBase | 用户管理的知识库容器 |
| DataSource | 数据来源配置，如 upload / website / oss |
| Document | 一个 URL、OSS object 或上传文件 |
| Chunk | Document 切分后的可检索单元 |
| IngestionJob | 一次导入/同步/重建任务 |
| IndexVersion | 可选，索引版本或构建批次 |
| AgentBinding | Agent 与知识库的绑定配置 |
| UserMetadata | 用户可编辑元数据 |
| SystemMetadata | 系统维护元数据 |

## 4. 存储设计

### 4.1 SQL Tables

#### `knowledge_bases`

```text
id                    varchar pk
name                  varchar
description           text
owner_user_id          varchar
visibility             varchar      # private | workspace | public
status                 varchar      # empty | processing | ready | has_errors | disabled
default_parser_config  json
default_retrieval_config json
embedding_config       json
vector_store_config    json
keyword_index_config   json
rerank_config          json nullable
active_index_version_id varchar nullable
document_count         int
chunk_count            int
created_at             datetime
updated_at             datetime
deleted_at             datetime nullable
```

索引：

- `(owner_user_id)`
- `(status)`
- `(deleted_at)`

#### `knowledge_data_sources`

```text
id                    varchar pk
kb_id                 varchar index
type                  varchar      # upload | website | oss
name                  varchar
config                json
sync_mode             varchar      # manual | scheduled
schedule              varchar nullable
status                varchar      # active | paused | error | deleted
last_sync_at           datetime nullable
last_error_code        varchar nullable
last_error_message     text nullable
created_by             varchar
created_at             datetime
updated_at             datetime
deleted_at             datetime nullable
```

`config` 示例：

```jsonc
// website
{
  "mode": "sitemap",
  "start_url": "https://docs.example.com",
  "sitemap_url": "https://docs.example.com/sitemap.xml",
  "include_patterns": ["/docs/*"],
  "exclude_patterns": ["/blog/*"],
  "max_pages": 200,
  "max_depth": 3,
  "respect_robots_txt": true
}
```

```jsonc
// oss
{
  "region": "cn-hangzhou",
  "bucket": "example-bucket",
  "prefix": "docs/",
  "file_types": [".pdf", ".md", ".html", ".txt"],
  "max_file_size_mb": 50,
  "credential_mode": "service_role"
}
```

#### `knowledge_documents`

```text
id                    varchar pk
kb_id                 varchar index
source_id             varchar nullable index
uri                   text         # URL / oss://bucket/key / upload://file_id
source_type            varchar
title                 varchar
description           text
mime_type              varchar
size_bytes             bigint
content_hash           varchar
etag                  varchar nullable
last_modified          datetime nullable
language               varchar nullable
tags                  json
category              varchar nullable
visibility             varchar
custom_metadata        json
system_metadata        json
status                 varchar      # queued | fetching | parsing | chunking | embedding | indexed | failed | disabled | deleted
chunk_count            int
error_code             varchar nullable
error_message          text nullable
indexed_at             datetime nullable
created_by             varchar
updated_by             varchar nullable
created_at             datetime
updated_at             datetime
deleted_at             datetime nullable
```

索引：

- `(kb_id, status)`
- `(kb_id, source_id)`
- `(kb_id, indexed_at)`
- `(kb_id, category)`
- `(source_type)`
- JSON tags 根据数据库能力后续优化。

系统元数据不可由用户编辑：

- `id`
- `uri`
- `content_hash`
- `etag`
- `source_id`
- `chunk_count`
- `indexed_at`
- `system_metadata`

用户可编辑元数据：

- `title`
- `description`
- `tags`
- `category`
- `visibility`
- `custom_metadata`

#### `knowledge_chunks`

```text
id                    varchar pk
kb_id                 varchar index
document_id            varchar index
chunk_index            int
text                  text
text_hash              varchar
heading_path           json
char_start             int nullable
char_end               int nullable
token_count            int
metadata               json
status                 varchar      # active | disabled | stale | deleted
embedding_ref          varchar nullable
indexed_at             datetime nullable
disabled_by            varchar nullable
disabled_reason        text nullable
created_at             datetime
updated_at             datetime
deleted_at             datetime nullable
```

约束：

- `(document_id, chunk_index)` unique
- disabled chunk 不进入检索结果。
- chunk 文本 MVP 不允许用户直接编辑。

#### `knowledge_ingestion_jobs`

```text
id                    varchar pk
kb_id                 varchar index
source_id             varchar nullable index
document_id            varchar nullable index
type                  varchar      # import | sync | reparse | rebuild_index | rebuild_chunks
trigger_type           varchar      # manual | schedule | conversation | system
triggered_by           varchar nullable
status                 varchar      # queued | running | completed | completed_with_errors | failed | cancelled
total_count            int
succeeded_count        int
failed_count           int
skipped_count          int
started_at             datetime nullable
finished_at            datetime nullable
error_summary          text nullable
created_at             datetime
updated_at             datetime
```

#### `knowledge_ingestion_job_items`

```text
id                    varchar pk
job_id                varchar index
document_id            varchar nullable
uri                   text
stage                 varchar      # discover | fetch | parse | chunk | embed | index
status                varchar      # queued | running | succeeded | failed | skipped
error_code             varchar nullable
error_message          text nullable
started_at             datetime nullable
finished_at            datetime nullable
```

#### `knowledge_index_versions`

```text
id                    varchar pk
kb_id                 varchar index
version               int
status                varchar      # building | active | failed | retired
embedding_provider_id  varchar
embedding_model        varchar
embedding_dimension    int
vector_store_provider_id varchar
vector_index_name      varchar
vector_namespace       varchar
keyword_index_name     varchar nullable
document_count         int
chunk_count            int
error_message          text nullable
created_by             varchar
created_at             datetime
activated_at           datetime nullable
retired_at             datetime nullable
```

用途：

- 记录每次索引构建使用的 embedding 和 vector store 配置。
- 支持后台构建新版本，完成后原子切换 `active_index_version_id`。
- embedding model、dimension、vector store provider、index name 变化时必须创建新版本并重建索引。
- MVP 可以只保留 active + building 两个版本，不做历史版本回滚。

#### `agent_knowledge_bindings`

```text
id                    varchar pk
agent_id               varchar index
kb_id                 varchar index
enabled                bool
top_k                 int
score_threshold        float
retrieval_mode         varchar      # hybrid | vector | keyword
force_citation         bool
query_rewrite_enabled  bool
created_by             varchar
created_at             datetime
updated_at             datetime
```

#### `knowledge_permissions`

```text
id                    varchar pk
kb_id                 varchar index
subject_type           varchar      # user | role | workspace
subject_id             varchar
permission             varchar      # view | query | edit | admin
created_by             varchar
created_at             datetime
```

### 4.2 File/Object Store

保存两类内容：

```text
knowledge/{kb_id}/raw/{document_id}/{original_filename}
knowledge/{kb_id}/normalized/{document_id}.md
```

原则：

- raw 用于重新解析。
- normalized markdown 用于全文 fetch 和重新 chunk。
- 删除文档优先软删除 SQL 状态，异步清理对象。

### 4.3 Vector Store

向量库只保存 active chunk 的 embedding 与检索 metadata。

向量 metadata：

```jsonc
{
  "kb_id": "kb_x",
  "document_id": "doc_x",
  "chunk_id": "chunk_x",
  "source_type": "website",
  "source_uri": "https://docs.example.com/a",
  "title": "Create EAS service",
  "tags": ["pai", "eas"],
  "category": "product-docs",
  "indexed_at": "2026-07-08T12:00:00Z"
}
```

### 4.4 Keyword Index

MVP 可选实现：

- SQLite FTS5
- Postgres `tsvector` + `pg_trgm`
- Elasticsearch/OpenSearch
- 简化版：SQL 文档候选 + Python exact match

推荐演进：

```text
MVP: SQL candidate + Python exact match
Phase 2: FTS/BM25
Phase 3: hybrid fusion + rerank
```

### 4.5 Configuration Model

配置分四层，避免把连接信息、检索策略和数据源参数混在一起：

```text
Global Provider Config     # admin 配置，系统级可用 provider
  -> KnowledgeBase Config  # 单个知识库选择 embedding/vector/index 策略
    -> DataSource Config   # 网站/OSS/上传的来源参数
      -> Agent Binding     # Agent 查询时的只读检索覆盖
```

优先级：

```text
AgentBinding retrieval override
  > KnowledgeBase retrieval defaults
  > Global defaults
```

限制：

- DataSource 不允许覆盖 embedding model 和 vector store，除非进入高级模式；否则同一知识库内会出现不可解释的检索质量差异。
- AgentBinding 只能覆盖 `top_k`、`score_threshold`、`retrieval_mode`、`force_citation`、`query_rewrite`、允许的 metadata filters。
- 普通用户只能查看必要的只读配置；provider 连接、密钥引用、索引重建只允许 admin/editor。

#### Global Provider Config

全局 provider 由 admin 配置。数据库或配置文件只保存 secret 引用，不保存明文 AK/SK/API Key。

```yaml
embedding_providers:
  - id: emb_dashscope_text_embedding_v3
    type: dashscope
    display_name: DashScope text-embedding-v3
    model: text-embedding-v3
    dimension: 1024
    normalize: true
    batch_size: 32
    timeout_seconds: 30
    api_key_secret_ref: DASHSCOPE_API_KEY

vector_store_providers:
  - id: vec_hologres_default
    type: hologres
    display_name: Hologres Default
    endpoint_secret_ref: HOLOGRES_ENDPOINT
    database: pai_rag
    table_prefix: kb_vectors
    credential_secret_ref: HOLOGRES_CREDENTIAL

keyword_index_providers:
  - id: kw_sql_default
    type: sql_exact_match
    display_name: SQL Exact Match

rerank_providers:
  - id: rerank_dashscope_default
    type: dashscope
    model: gte-rerank
    api_key_secret_ref: DASHSCOPE_API_KEY
```

Provider 需要提供两个标准动作：

```text
test_connection(provider_id)
describe_capabilities(provider_id)
```

`describe_capabilities` 返回：

```jsonc
{
  "provider_id": "emb_dashscope_text_embedding_v3",
  "kind": "embedding",
  "models": [
    {
      "name": "text-embedding-v3",
      "dimension": 1024,
      "max_batch_size": 64,
      "max_input_tokens": 8192
    }
  ]
}
```

#### KnowledgeBase Index Config

创建知识库时必须选择 embedding provider 和 vector store provider；可以用系统默认值自动填充。

```jsonc
{
  "embedding_config": {
    "provider_id": "emb_dashscope_text_embedding_v3",
    "model": "text-embedding-v3",
    "dimension": 1024,
    "normalize": true,
    "batch_size": 32
  },
  "vector_store_config": {
    "provider_id": "vec_hologres_default",
    "index_name": "kb_kb_pai_v1",
    "namespace": "kb_pai",
    "metric": "cosine",
    "dimension": 1024
  },
  "keyword_index_config": {
    "provider_id": "kw_sql_default",
    "index_name": "kb_keyword_kb_pai_v1",
    "enabled": true
  },
  "rerank_config": {
    "enabled": false,
    "provider_id": "rerank_dashscope_default",
    "model": "gte-rerank",
    "top_n": 20
  },
  "default_parser_config": {
    "chunk_size": 1000,
    "chunk_overlap": 150
  },
  "default_retrieval_config": {
    "mode": "hybrid",
    "top_k": 6,
    "score_threshold": 0.25,
    "force_citation": true
  }
}
```

校验规则：

- `embedding_config.dimension` 必须等于 `vector_store_config.dimension`。
- `vector_store_config.metric` 需要和 embedding normalize 策略匹配；默认 `normalize=true + cosine`。
- `index_name`、`namespace` 必须由后端生成或校验，避免用户构造跨租户索引名。
- provider 不可用时，知识库状态进入 `has_errors`，但不删除已有索引。

#### Config Change Policy

不同配置变更影响不同：

| 配置项 | 是否需要重建 | 处理方式 |
|--------|--------------|----------|
| `top_k` / `score_threshold` | 否 | 立即生效 |
| `retrieval.mode` | 否 | 立即生效，前提是相关索引存在 |
| `chunk_size` / `chunk_overlap` | 是 | 触发 `rebuild_chunks`，重新 embed 和索引 |
| `embedding.model` / `dimension` | 是 | 创建新 `IndexVersion`，全量 re-embed |
| `vector_store.provider_id` | 是 | 创建新 `IndexVersion`，全量写入新向量库 |
| `vector_store.index_name` / `namespace` | 是 | 创建新 `IndexVersion` |
| `keyword_index.provider_id` | 是 | 重建 keyword index，可不重建 vector |
| `rerank.enabled` / `top_n` | 否 | 查询链路立即生效 |

配置更新流程：

```text
PATCH index config
  -> validate provider and dimension
  -> if runtime-only change: save config
  -> if index-affecting change: create rebuild_index job
  -> build new index version in background
  -> smoke test retrieval
  -> switch active_index_version_id
  -> retire previous version
```

#### Vector Store Choices

推荐按部署规模选择：

| 场景 | 推荐 |
|------|------|
| 本地开发 / 单机 PoC | SQLite/FAISS 或 SQL exact match + mock vector |
| 小规模生产 | Postgres pgvector / Hologres / AnalyticDB |
| 大规模生产 | Milvus / OpenSearch / Elasticsearch / Hologres |
| 阿里云优先部署 | Hologres、AnalyticDB、OpenSearch 或已有 PAI-RAG 支持的向量后端 |

设计约束：

- Retrieval Service 只依赖统一 `VectorStoreAdapter`，不要在业务代码里直接写具体向量库 SDK。
- 向量库中保存最小 metadata；完整文档、权限、状态仍以 SQL 为准。
- 查询结果必须回 SQL 二次校验权限和 chunk/document 状态。

## 5. 离线导入流程

### 5.1 Pipeline

```text
trigger job
  -> discover
  -> fetch
  -> store raw
  -> parse to normalized markdown
  -> chunk
  -> embed
  -> write vector index
  -> write keyword index
  -> mark document indexed
  -> finalize job
```

### 5.2 Trigger Sources

触发方式：

- UI 手动导入。
- UI 手动同步。
- 对话管理工具触发。
- 定时同步。
- 管理 API 触发。
- 系统重建索引触发。

### 5.3 Job Queue

建议先实现轻量后台任务抽象，后续可切 Celery/RQ/Arq。

接口：

```python
class JobQueue:
    async def enqueue(job_type: str, payload: dict) -> str: ...
    async def cancel(job_id: str) -> bool: ...
```

MVP 如果不引入独立 worker，可先用进程内 `asyncio.create_task`，但需要接受：

- 服务重启会丢任务。
- 无法横向扩容。
- 长任务会影响 API 进程。

推荐生产方案：

- Redis + Celery/RQ/Arq。
- 单独 ingestion worker。
- 定时任务由 scheduler 触发。

### 5.4 Website Adapter

阶段：

```text
discover URLs
  -> normalize URL
  -> filter include/exclude
  -> fetch HTML
  -> extract main content
  -> convert to markdown
```

MVP 支持：

- single URL
- sitemap

Limited crawl 可作为 Phase 1.5。

安全要求：

- 禁止访问私网/localhost，除非管理员显式允许。
- 限制跳转次数。
- 限制页面大小。
- 限速。
- 尊重 robots.txt 默认开启。

### 5.5 OSS Adapter

阶段：

```text
list objects
  -> filter prefix/type/size
  -> compare etag/last_modified
  -> download changed objects
  -> parse
```

凭证模式：

- `service_role`: 服务端配置的统一访问角色。
- `user_ram_role`: 用户授权的 RAM Role，按用户或租户访问 OSS。

不允许：

- 前端传 AK/SK。
- 明文保存 AK/SK 到数据库。

## 6. API Design

API 分三层：

```text
Management API     # admin/editor
Query API          # user/agent
Job API            # status/control
```

### 6.1 Management API

仅 admin/editor 使用。普通用户默认无权调用写接口。

#### KnowledgeBase

```http
GET    /v1/knowledge-bases
POST   /v1/knowledge-bases
GET    /v1/knowledge-bases/{kb_id}
PATCH  /v1/knowledge-bases/{kb_id}
DELETE /v1/knowledge-bases/{kb_id}
```

Create request：

```jsonc
{
  "name": "PAI Docs",
  "description": "PAI product documentation",
  "visibility": "workspace",
  "embedding_config": {
    "provider_id": "emb_dashscope_text_embedding_v3",
    "model": "text-embedding-v3",
    "dimension": 1024
  },
  "vector_store_config": {
    "provider_id": "vec_hologres_default",
    "metric": "cosine"
  },
  "keyword_index_config": {
    "provider_id": "kw_sql_default",
    "enabled": true
  },
  "rerank_config": {
    "enabled": false
  },
  "default_parser_config": {
    "chunk_size": 1000,
    "chunk_overlap": 150
  },
  "default_retrieval_config": {
    "mode": "hybrid",
    "top_k": 6,
    "score_threshold": 0.25
  }
}
```

#### Provider Config

仅 admin 可用。用于维护 embedding、vector store、keyword index、rerank provider。

```http
GET    /v1/knowledge/providers
POST   /v1/knowledge/providers
GET    /v1/knowledge/providers/{provider_id}
PATCH  /v1/knowledge/providers/{provider_id}
DELETE /v1/knowledge/providers/{provider_id}
POST   /v1/knowledge/providers/{provider_id}/test-connection
GET    /v1/knowledge/providers/{provider_id}/capabilities
```

Provider create request：

```jsonc
{
  "kind": "embedding",
  "type": "dashscope",
  "display_name": "DashScope text-embedding-v3",
  "config": {
    "model": "text-embedding-v3",
    "dimension": 1024,
    "normalize": true,
    "batch_size": 32,
    "api_key_secret_ref": "DASHSCOPE_API_KEY"
  }
}
```

返回体必须隐藏 secret value：

```jsonc
{
  "id": "emb_dashscope_text_embedding_v3",
  "kind": "embedding",
  "type": "dashscope",
  "display_name": "DashScope text-embedding-v3",
  "config": {
    "model": "text-embedding-v3",
    "dimension": 1024,
    "api_key_secret_ref": "DASHSCOPE_API_KEY",
    "secret_status": "configured"
  }
}
```

#### KnowledgeBase Index Config

仅 admin/editor 可用。用于配置单个知识库使用的 embedding、vector store、keyword index 和 rerank 策略。

```http
GET   /v1/knowledge-bases/{kb_id}/index-config
PATCH /v1/knowledge-bases/{kb_id}/index-config
POST  /v1/knowledge-bases/{kb_id}/index-config/validate
POST  /v1/knowledge-bases/{kb_id}/index/rebuild
GET   /v1/knowledge-bases/{kb_id}/index/status
GET   /v1/knowledge-bases/{kb_id}/index/versions
```

Patch request：

```jsonc
{
  "embedding_config": {
    "provider_id": "emb_dashscope_text_embedding_v3",
    "model": "text-embedding-v3",
    "dimension": 1024
  },
  "vector_store_config": {
    "provider_id": "vec_hologres_default",
    "metric": "cosine"
  },
  "rerank_config": {
    "enabled": true,
    "provider_id": "rerank_dashscope_default",
    "top_n": 20
  }
}
```

Patch response：

```jsonc
{
  "kb_id": "kb_pai",
  "requires_rebuild": true,
  "rebuild_reason": [
    "embedding model changed",
    "vector dimension changed"
  ],
  "job_id": "job_rebuild_x",
  "active_index_version_id": "idx_v1",
  "building_index_version_id": "idx_v2"
}
```

#### DataSource

```http
GET    /v1/knowledge-bases/{kb_id}/sources
POST   /v1/knowledge-bases/{kb_id}/sources
GET    /v1/knowledge-bases/{kb_id}/sources/{source_id}
PATCH  /v1/knowledge-bases/{kb_id}/sources/{source_id}
DELETE /v1/knowledge-bases/{kb_id}/sources/{source_id}
POST   /v1/knowledge-bases/{kb_id}/sources/{source_id}/sync
POST   /v1/knowledge-bases/{kb_id}/sources/{source_id}/pause
POST   /v1/knowledge-bases/{kb_id}/sources/{source_id}/resume
POST   /v1/knowledge-bases/{kb_id}/sources/{source_id}/preview
POST   /v1/knowledge-bases/{kb_id}/sources/{source_id}/test-connection
```

#### Documents

```http
GET    /v1/knowledge-bases/{kb_id}/documents
GET    /v1/knowledge-bases/{kb_id}/documents/{document_id}
PATCH  /v1/knowledge-bases/{kb_id}/documents/{document_id}/metadata
POST   /v1/knowledge-bases/{kb_id}/documents/{document_id}/retry
POST   /v1/knowledge-bases/{kb_id}/documents/{document_id}/reparse
POST   /v1/knowledge-bases/{kb_id}/documents/{document_id}/disable
POST   /v1/knowledge-bases/{kb_id}/documents/{document_id}/enable
DELETE /v1/knowledge-bases/{kb_id}/documents/{document_id}
```

Document list filters：

```text
source_id
source_type
status
mime_type
tag
category
created_by
indexed_from
indexed_to
updated_from
updated_to
query
```

Metadata patch：

```jsonc
{
  "title": "EAS Billing",
  "description": "Billing rules for EAS",
  "tags": ["pai", "eas", "billing"],
  "category": "billing",
  "visibility": "workspace",
  "custom_metadata": {
    "product": "PAI",
    "module": "EAS"
  }
}
```

#### Chunks

```http
GET  /v1/knowledge-bases/{kb_id}/documents/{document_id}/chunks
GET  /v1/knowledge-bases/{kb_id}/chunks/{chunk_id}
POST /v1/knowledge-bases/{kb_id}/chunks/{chunk_id}/disable
POST /v1/knowledge-bases/{kb_id}/chunks/{chunk_id}/enable
POST /v1/knowledge-bases/{kb_id}/documents/{document_id}/chunks/rebuild
POST /v1/knowledge-bases/{kb_id}/chunks/rebuild
```

Chunk list filters：

```text
document_id
status
query
heading
min_tokens
max_tokens
```

MVP 不提供：

```http
PATCH /chunks/{chunk_id}/text
```

#### Agent Bindings

```http
GET    /v1/agents/{agent_id}/knowledge-bindings
POST   /v1/agents/{agent_id}/knowledge-bindings
PATCH  /v1/agents/{agent_id}/knowledge-bindings/{binding_id}
DELETE /v1/agents/{agent_id}/knowledge-bindings/{binding_id}
```

### 6.2 Query API

普通用户和 Agent 都可使用，但必须检查：

- 用户是否可 query 该知识库。
- 该知识库是否绑定给当前 Agent。
- disabled document/chunk 不返回。

#### Semantic Search

```http
POST /v1/knowledge/query/search
```

```jsonc
{
  "kb_ids": ["kb_pai"],
  "query": "如何创建 EAS 服务？",
  "top_k": 6,
  "score_threshold": 0.25,
  "mode": "hybrid",
  "filters": {
    "source_type": "website",
    "tags": ["eas"],
    "category": "product-docs",
    "date_from": "2026-01-01",
    "date_to": "2026-07-08"
  }
}
```

#### Keyword Match

```http
POST /v1/knowledge/query/keyword
```

```jsonc
{
  "kb_ids": ["kb_pai"],
  "pattern": "InvalidParameter",
  "is_regex": false,
  "case_sensitive": false,
  "filters": {
    "source_type": "oss"
  },
  "limit": 20,
  "context_lines": 2
}
```

#### Title / Catalog Search

```http
POST /v1/knowledge/query/catalog
```

```jsonc
{
  "kb_ids": ["kb_pai"],
  "query": "EAS 计费",
  "search_fields": ["title", "uri", "tags", "category"],
  "filters": {
    "status": "indexed",
    "source_type": "website"
  },
  "limit": 20
}
```

#### Date / Metadata Search

```http
POST /v1/knowledge/query/metadata
```

```jsonc
{
  "kb_ids": ["kb_pai"],
  "filters": {
    "indexed_from": "2026-07-01",
    "indexed_to": "2026-07-08",
    "tags": ["billing"],
    "mime_type": "application/pdf"
  },
  "sort": {
    "field": "indexed_at",
    "order": "desc"
  },
  "limit": 50
}
```

#### Fetch Full Document / Chunk Context

```http
POST /v1/knowledge/query/fetch
```

```jsonc
{
  "kb_id": "kb_pai",
  "ref": {
    "document_id": "doc_x"
  },
  "mode": "full_doc",
  "max_chars": 6000,
  "offset": 0
}
```

Modes：

- `full_doc`
- `chunk`
- `chunk_neighbors`
- `section`

### 6.3 Job API

```http
GET  /v1/knowledge/jobs
GET  /v1/knowledge/jobs/{job_id}
POST /v1/knowledge/jobs/{job_id}/cancel
POST /v1/knowledge/jobs/{job_id}/retry-failed
GET  /v1/knowledge/jobs/{job_id}/items
```

## 7. Agent Tool Design

Agent 工具分两类：

```text
Runtime query tools      # 普通用户可用，只读
Admin management tools   # 管理员可用，可写
```

### 7.1 Runtime Query Tools

这些工具可以暴露给普通 Agent。

#### `knowledge_search`

语义/混合检索。

```jsonc
{
  "query": "如何创建 EAS 服务？",
  "kb_ids": ["kb_pai"],
  "top_k": 6,
  "filters": {
    "tags": ["eas"]
  }
}
```

#### `knowledge_keyword_match`

精确关键词、错误码、配置项查找。

```jsonc
{
  "pattern": "InvalidParameter",
  "kb_ids": ["kb_pai"],
  "case_sensitive": false,
  "context_lines": 2
}
```

#### `knowledge_catalog_search`

按标题、路径、标签、分类找文档。

```jsonc
{
  "query": "EAS 计费",
  "kb_ids": ["kb_pai"],
  "limit": 10
}
```

#### `knowledge_metadata_search`

按日期、来源、类型、标签筛选文档。

```jsonc
{
  "kb_ids": ["kb_pai"],
  "filters": {
    "indexed_from": "2026-07-01",
    "source_type": "oss"
  }
}
```

#### `knowledge_fetch`

取全文、chunk 邻域、章节。

```jsonc
{
  "kb_id": "kb_pai",
  "document_id": "doc_x",
  "mode": "chunk_neighbors",
  "chunk_id": "chunk_x",
  "window": 1
}
```

Agent 使用约束：

- 工具只看到当前 Agent 绑定的知识库。
- 工具只返回 active/indexed 内容。
- 默认返回带来源引用的结果。
- `fetch` 必须有 `max_chars` 截断，避免撑爆上下文。

### 7.2 Admin Management Tools

这些工具只允许 admin/operator Agent 使用。

#### `create_knowledge_base`

创建知识库。

#### `add_knowledge_source`

添加 website / oss / upload source。

#### `sync_knowledge_source`

触发同步。

#### `retry_failed_documents`

重试失败文档。

#### `edit_document_metadata`

编辑文档用户元数据。

#### `disable_document`

禁用文档。

#### `disable_chunk`

禁用 chunk。

#### `rebuild_document_chunks`

重建单文档 chunks。

#### `bind_knowledge_base_to_agent`

绑定知识库到 Agent。

管理工具安全要求：

- 必须走 `require_admin` 或等价权限。
- 删除、全量重建、批量修改必须二次确认。
- 所有写操作进入 audit log。
- 工具返回必须包含 job id 或 resource id。

## 8. Frontend UI

### 8.1 Knowledge List

入口：Settings 或主导航新增 `Knowledge`。

展示：

- 知识库状态。
- 文档数。
- chunk 数。
- 数据源数。
- 最近同步。
- 已绑定 Agent。

### 8.2 Knowledge Detail

Tabs：

```text
Overview
Sources
Documents
Metadata
Chunks
Retrieval Test
Agent Binding
Settings
```

Settings 包含：

- `Indexing & Retrieval`：embedding provider、vector store provider、keyword index、rerank、parser、默认 retrieval 参数。
- `Permissions`：知识库可见性、query/editor/admin 权限。
- `Danger Zone`：禁用知识库、删除知识库、全量重建。

普通用户进入知识库详情时：

- 可看知识库名称、描述、可查询状态、绑定 Agent 状态。
- 不展示 provider secret、vector store endpoint、索引名等内部连接配置。
- 不允许触发同步、重建、删除、元数据编辑。

管理员/editor 在 `Indexing & Retrieval` 可操作：

- 选择 embedding provider 和 model。
- 选择 vector store provider。
- 查看 dimension、metric、namespace、active index version。
- 测试 embedding provider 连接。
- 测试 vector store provider 连接。
- 校验配置兼容性。
- 触发全量 rebuild index。
- 调整默认 top_k、score threshold、retrieval mode。

配置变更提示：

```text
Changing embedding model or vector store requires rebuilding the index.
Existing search will keep using the active index until rebuild succeeds.
```

### 8.3 Sources Tab

支持：

- 新增 website source。
- 新增 OSS source。
- 上传文件 source。
- 连接测试。
- 预览待导入对象。
- 手动同步。
- 暂停/恢复 source。

### 8.4 Documents Tab

支持：

- 多条件筛选。
- 元数据编辑 drawer。
- Retry / Re-parse / Delete / Disable。
- 查看 chunk preview。

### 8.5 Metadata Tab

支持：

- 元数据筛选。
- 批量加标签。
- 批量分类。
- 导出 CSV。

### 8.6 Chunks Tab

支持：

- 按文档查看 chunks。
- 搜索 chunk text。
- 禁用/恢复 chunk。
- 单文档重建 chunks。

不支持：

- 直接编辑 chunk text。

### 8.7 Retrieval Test

子模式：

- Semantic
- Keyword
- Catalog
- Metadata
- Fetch

每个测试结果都显示：

- 命中内容。
- score。
- source。
- document。
- chunk id。
- 是否会被 Agent 使用。

### 8.8 Agent Binding

支持：

- 选择 Agent。
- 配置 top_k / threshold / mode。
- 开启强制引用。
- 开启 query rewrite。

## 9. Permission Model

### 9.1 Roles

| Role | Capabilities |
|------|--------------|
| Admin | 全局管理、配置、删除、绑定 Agent |
| Editor | 维护有权限知识库、同步、元数据编辑、chunk 禁用 |
| User | 查询有权限知识库 |

### 9.2 Query Permission

普通用户查询需要满足：

```text
user has query permission on kb
AND kb is enabled
AND document status = indexed
AND chunk status = active
```

Agent 查询需要额外满足：

```text
kb bound to current agent
AND binding enabled
```

### 9.3 Management Permission

写操作需要：

```text
admin OR editor permission on kb
```

全局配置和 Agent 绑定需要：

```text
admin
```

## 10. Offline Worker Reliability

### 10.1 Idempotency

每个 document upsert 以：

```text
kb_id + source_id + uri
```

作为稳定键。

如果 `content_hash` 未变化：

- 跳过 parse/chunk/embed。
- 只更新 metadata 和 last_seen。

### 10.2 Retry

失败 item 支持单独 retry。

Retry 策略：

```text
network errors: exponential backoff
parse errors: manual retry only unless parser changed
embedding errors: retryable
permission errors: manual fix required
```

### 10.3 Cancellation

Job cancel 后：

- 未开始 item 标记 cancelled。
- 正在执行 item 尽量协作停止。
- 已完成 item 保留。
- KnowledgeBase 状态重新计算。

### 10.4 Rebuild

重建类型：

- `reparse_document`: raw -> normalized -> chunks -> embeddings
- `rebuild_chunks`: normalized -> chunks -> embeddings
- `rebuild_embeddings`: chunks -> embeddings
- `rebuild_keyword_index`: chunks -> keyword index

## 11. Observability

必须记录：

- 谁创建了知识库。
- 谁添加了 source。
- 谁触发了 job。
- 谁编辑了 metadata。
- 谁禁用了 document/chunk。
- Agent 使用了哪些 knowledge chunks。

建议新增：

```text
knowledge_audit_logs
```

字段：

```text
id
actor_user_id
action
resource_type
resource_id
before
after
created_at
```

## 12. MVP Implementation Plan

### Phase 1: Storage and Management API

- Add SQL tables。
- Add file/object storage abstraction。
- Add CRUD APIs。
- Add document metadata edit。
- Add chunk list/disable/rebuild API。

### Phase 2: Ingestion Pipeline

- Upload source。
- Website single URL + sitemap adapter。
- OSS prefix adapter。
- Ingestion job status。
- Manual sync。
- Retry failed。

### Phase 3: Retrieval API and Agent Tools

- `knowledge_search`
- `knowledge_keyword_match`
- `knowledge_catalog_search`
- `knowledge_metadata_search`
- `knowledge_fetch`
- Agent binding enforcement。

### Phase 4: Frontend

- Knowledge list/detail。
- Sources/Documents/Metadata/Chunks tabs。
- Retrieval test。
- Agent binding UI。

### Phase 5: Admin Tools and Conversation Control

- Admin management tools。
- Conversation confirmations。
- Task result cards。
- Audit logs。

## 13. Open Decisions

- 默认 embedding provider 使用 DashScope，还是允许部署方通过环境变量指定？
- 默认 vector store 使用现有 PAI-RAG 后端中最易部署的一种，还是支持 admin 首次启动时选择？
- 是否允许用户自带 vector store 连接？建议 MVP 不开放，只允许平台 admin 配置。
- IndexVersion 是否需要支持回滚？建议 MVP 只支持 active/building/retired，不暴露回滚入口。
- Provider 配置存储在数据库还是部署配置文件？建议 MVP 使用环境变量 + 后端配置，Phase 2 再做 UI 管理。
- Keyword index MVP 用 SQL + Python scan，还是直接引入 FTS5/PG trigram？
- OSS 凭证默认使用 service role，还是优先复用用户 RAM 授权？
- Editor 角色是否需要在当前账号系统中实现，还是先 admin/user 两级？
- `knowledge_metadata_search` 是否作为独立 Agent 工具，还是合并进 `catalog_search`？
- Chunk disable 是否 MVP 必做，还是先只做 chunk view + rebuild？
- 是否要支持 document-level visibility，还是先只支持 KB-level permission？

## 14. Recommended Defaults

```yaml
embedding:
  provider_id: emb_dashscope_text_embedding_v3
  model: text-embedding-v3
  dimension: 1024
  normalize: true
  batch_size: 32
vector_store:
  provider_id: vec_default
  metric: cosine
  namespace_strategy: kb_id
keyword_index:
  enabled: true
  provider_id: kw_sql_default
rerank:
  enabled: false
parser:
  chunk_size: 1000
  chunk_overlap: 150
retrieval:
  mode: hybrid
  top_k: 6
  score_threshold: 0.25
  force_citation: true
website:
  max_pages: 200
  max_depth: 3
  respect_robots_txt: true
oss:
  max_file_size_mb: 50
fetch:
  max_chars: 6000
```

## 15. Non-goals for MVP

- 自由编辑 chunk text。
- 文档级 ACL。
- 多版本索引回滚。
- OCR。
- 动态浏览器网页渲染。
- 自动质量评估。
- 多 workspace。

## 16. Summary

推荐第一版按以下闭环实现：

```text
Knowledge CRUD
  -> Source import
  -> Offline ingestion job
  -> Document metadata management
  -> Chunk view/rebuild/disable
  -> Retrieval test
  -> Agent binding
  -> Agent runtime query tools
```

普通用户只获得查询能力；管理员和授权 editor 才能管理知识库。Agent 工具必须分为只读 runtime tools 与 admin management tools，避免普通对话通过 prompt 越权修改知识库。
