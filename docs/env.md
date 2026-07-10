
# 环境变量说明
### 1. 使用CUDA加速

| 名称 | 取值 | 说明 |
| - | - | - |
| use_cuda | `true`/`false` | 是否使用CUDA加速，默认为false（开启后本地embedding模型和pdf解析效率会大大提升）|

### 2.数据库配置

PAI-RAG 支持三种数据库类型：SQLite（默认）、PostgreSQL 和 MySQL。

**SQLite（默认）**
- 适合开发、测试和小规模部署场景
- 数据保存在本地，不支持多实例访问
- 配置方式：无需额外配置，或设置 `DB_TYPE=sqlite`

**PostgreSQL**
- 推荐用于生产环境，提供更好的并发性能和事务支持
- 配置方式：设置 `DB_TYPE=postgresql` 并填写以下连接信息

**MySQL**
- 适合需要与现有 MySQL 基础设施集成的场景
- **重要：MySQL 数据库必须使用 utf8mb4 编码**
- 配置方式：设置 `DB_TYPE=mysql` 并填写以下连接信息

| 名称 | 取值 | 说明 |
| - | - | - |
| DB_TYPE | enum, `sqlite`/`postgresql`/`mysql` | 默认为`sqlite`，`postgresql` 和 `mysql` 需要填写下面的连接信息 |
| DB_HOST | STRING | HOST 地址，推荐使用VPC内网直连，默认为 `localhost` |
| DB_PORT | STRING | 端口，PostgreSQL 默认为 `5432`，MySQL 默认为 `3306` |
| DB_USER | STRING | 用户名 |
| DB_PASSWORD | STRING | 密码 | 
| DB_NAME | STRING | 数据库名称 |

**MySQL 数据库创建示例：**
```sql
CREATE DATABASE your_database_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```


### 3. 文件存储配置
知识库文件/图片默认存在本地目录，如果需要持久化/有多模态图片理解需求，需要配置OSS存储地址：
| 名称 | 取值 | 说明 |
| - | - | - |
| FILE_STORE_TYPE | enum, `local`/`oss` | 默认为本地，`oss`需填写下面的信息 |
| OSS_BUCKET | | OSS BUCKET名称 |
| OSS_ENDPOINT | | OSS endpoint,如oss-cn-hangzhou.aliyuncs.com |
| OSS_ACCESS_KEY_ID | | 有OSS BUCKET权限的AK |
| OSS_ACCESS_KEY_SECRET | | 有OSS BUCKET权限的SK |

### 4. 向量检索引擎配置

推荐使用Aliyun Milvus / Elasticsearch / PostgreSQL 向量检索引擎，默认为本地chroma（数据保存在本地，不支持多实例访问，重启会失效，挂载OSS目录可以实现持久化存储）。

| 名称 | 取值 | 说明 |
| - | - | - |
| VECTOR_DB_TYPE | enum, `local`/`elasticsearch`/`milvus`/`postgresql` | 默认为`local`，如果选择其它类型请配置如下连接信息 |


- **Milvus**

| 名称 | 取值 | 说明 |
| - | - | - |
| MILVUS_HOST | STRING | HOST 地址，推荐使用VPC内网直连 |
| MILVUS_PORT | STRING | 端口，默认为19530 |
| MILVUS_USER | STRING | 用户名，如root |
| MILVUS_PASSWORD | STRING | 密码 | 
| MILVUS_DATABASE | STRING | 数据库名称，如default |


- **ElasticSearch**

| 名称 | 取值 | 说明 |
| - | - | - |
| ELASTICSEARCH_URL | STRING | 服务地址，如https://eas_host:9200 |
| ELASTICSEARCH_USER | STRING | 用户名，如elastic |
| ELASTICSEARCH_PASSWORD | STRING | 密码 | 


- **PostgreSQL**

| 名称 | 取值 | 说明 |
| - | - | - 
| POSTGRES_HOST | STRING | HOST 地址，推荐使用VPC内网直连 |
| POSTGRES_PORT | STRING | 端口，默认为5432 |
| POSTGRES_USER | STRING | 用户名 |
| POSTGRES_PASSWORD | STRING | 密码 | 
| POSTGRES_DATABASE | STRING | 数据库名称 |


### 5. 文件产物分发（Artifact）

Agent 在沙箱 `/mnt/user` 下生成的文件（报告、图表、HTML 等），可通过 `publish_artifact` 工具在前端预览或下载。该能力默认关闭，**必须设置 `FILES_URL_SECRET` 才启用**（为空时 `publish_artifact` 直接拒绝，属于 fail-closed）。

| 名称 | 取值 | 说明 |
| - | - | - |
| FILES_URL_SECRET | STRING | 签发 `/v1/files` 访问令牌的 HMAC 密钥。为空则文件分发功能关闭。请设置为足够长的随机串 |
| FILES_NAS_LOCAL_ROOT | STRING | 后端主机上挂载的、与沙箱 `/mnt/user` 同一份 NAS 导出的本地路径。设置后直接读该挂载分发文件；留空则回退为从活动沙箱读回字节 |
| FILES_MAX_BYTES | INT | 可内联预览的单文件大小上限，默认 `26214400`（25 MiB）。超限文件仍可下载，但不内联预览 |

### 6. 可观测性（OpenTelemetry / Langfuse 追踪）

记录 AgentLoop 的每一步——大模型推理（输入/输出/token）与工具输入/输出——为一棵嵌套 trace，走 OpenInference 语义约定，可被 Langfuse、Arize Phoenix、Jaeger 等原生消费。**不设置任何变量即为关闭**（零开销）。二选一：

**方式 A — 标准 OTLP**（任意 OpenTelemetry Collector / Jaeger / Tempo）：

| 名称 | 取值 | 说明 |
| - | - | - |
| OTEL_EXPORTER_OTLP_ENDPOINT | STRING | OTLP 服务地址（含端口），如 `http://otel-collector:4318` |
| OTEL_EXPORTER_OTLP_HEADERS | STRING | 鉴权头，`key=value` 逗号分隔，如 `Authorization=Basic <base64>` |
| OTEL_EXPORTER_OTLP_PROTOCOL | enum, `http/protobuf`/`grpc` | 传输协议，默认 `http/protobuf`（http 会自动补 `/v1/traces`） |
| OTEL_SERVICE_NAME | STRING | 服务名，默认 `pai-agent` |
| OTEL_TRACES_SAMPLER_ARG | FLOAT | 采样率 0..1，默认 `1.0`（全采） |
| OTEL_TRACES_ENABLED | enum, `auto`/`true`/`false` | 总开关，默认 `auto`（解析到端点即启用） |

**方式 B — Langfuse**（自动推导上面的 endpoint 与 Basic 鉴权头）：

| 名称 | 取值 | 说明 |
| - | - | - |
| LANGFUSE_HOST | STRING | Langfuse 地址，默认 `https://cloud.langfuse.com` |
| LANGFUSE_PUBLIC_KEY | STRING | Public Key（`pk-lf-...`） |
| LANGFUSE_SECRET_KEY | STRING | Secret Key（`sk-lf-...`），endpoint = `{HOST}/api/public/otel` |

> 注意：真正导出需运行环境已安装 `opentelemetry` / `openinference` 相关依赖（`poetry install`）；未安装时代码走 fallback，不报错但也不产出 span。

