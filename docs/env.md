
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

