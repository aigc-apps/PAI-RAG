# 知识库 DataSource 层 — 实现说明

> 配套设计契约见 [`datasource-retrieval-spec.md`](./datasource-retrieval-spec.md)。本文记录该契约在 PAI-RAG 中的**落地实现**:数据模型、适配器、增量同步、4 个 Agent 检索工具、REST API、前端、以及关键设计决策与兼容性。

## 1. 总览

为知识库(KB)新增一层**数据源(DataSource)**:一个数据源**归属于一个 KB**(KB → 0..N 数据源),从外部站点抓取文档,**复用现有入库管道**(FileParser → 切片 → 嵌入 → 向量库)按 KB 的 `chunk_config`/`embedding` 解析存储,并给 Agent 暴露 **search / catalog / keyword / fetch** 四个正交检索工具。

核心原则:**复用**(抓取文档走现有 KbFile/KbChunk 管道,零新增解析/嵌入代码)、**源无关**(适配器归一,下游解耦)、**增量**(manifest + content_hash diff)。

---

## 2. 数据模型

三张新表(`backend/db/models/knowledgebase/datasource.py`),枚举在 `backend/common/knowledgebase/types.py`:

| 表 | 作用 | 关键字段 |
|---|---|---|
| `pai_datasource` | 配置 + 聚合态 | `kb_id`(FK,CASCADE)、`datasource_key`(KB 内唯一)、`source_type`、`source_config`(JSON)、`sync_schedule`、`next_sync_at`、`enabled`、`status`、`last_sync_at/finished_at/duration_ms`、`doc_count`、`last_sync_report`、`last_error` |
| `pai_datasource_document` | 文档清单(manifest = 数据源内"文件列表") | `doc_id`(`VARCHAR(512)`,`{key}/{path}`)、`file_id`(→KbFile,回填)、`path`、`source_url`、`fetch_url`、`title`、`section`、`product`、`summary`、`lang`、`content_hash`、`byte_size`、`source_meta`(JSON)、`doc_status`、时间线 |
| `pai_datasource_sync_run` | 同步历史 | `trigger`、`triggered_by`、`status`、`started/finished_at`、`duration_ms`、`n_discovered/added/updated/deleted/unchanged/failed`、`report` |

唯一约束:`(kb_id, datasource_key, tenant_id)`、`(datasource_id, doc_id)`;索引 `(datasource_id, doc_status)`。

**枚举**:`DataSourceType` = `llms_txt | sphinx | local | github`(后两者预留未注册);`DataSourceStatus` = `idle | syncing | ingesting | succeeded | partial | failed | cancelled`;`DataSourceDocStatus` = `discovered | fetching | ingesting | synced | failed | cancelled | deleted`;`SyncRunStatus`、`SyncTrigger`。

**KbFileEntity 打标(不加列,用 `file_metadata` JSON + `file_source`)**:抓取的文档入库为普通 `KbFileEntity`,`file_metadata` 携带 `title / source_url / source_site / summary / product / section / lang / datasource_id / datasource_key / source_doc_id / fetched_from / content_hash`,`file_source` = 可点击 URL。`file_metadata` 会被切片服务并入每个 chunk 的 metadata,因此这些字段**同时驱动检索过滤、搜索结果、文件展示**。`doc_id == file_id` 不变,spec 的 doc_id 存为 `source_doc_id`。

> 表由 `SQLModel.metadata.create_all` 自动建(无 Alembic),新实体已在 `backend/db/models/__init__.py` 注册。

---

## 3. 适配器框架

`backend/rag/datasource/`:`schema.py`(`DiscoveredDoc` / `SourceDocument`)、`base_adapter.py`(`BaseAdapter` ABC:抽象 `discover`/`fetch`,**源无关 `emit`**:算 content_hash、doc_id、source_site、剥 frontmatter、相对链接→绝对、**title 兜底**=适配器标题→正文首个 markdown 标题→文件名)、`registry.py`、`http_util.py`、`url_guard.py`。

| 适配器 | source_config | discover / fetch |
|---|---|---|
| `llms_txt` | `product` 或 `llms_url`,`lang?`,`sections?` | 拉 llms.txt 清单 → 下载官方 `.md` |
| `sphinx` | `base_url`(必填),`product?`,`lang?`,`workers?` | toctree 种子 + **BFS** 跟随同站链接;HTML 正文区 → markdownify;BFS 不完整时置 `discovery_partial` |

**SSRF 防护**(`url_guard.py`,所有适配器 HTTP 经 `http_util.http_get`):限 http/https、解析并拒绝私网/回环/链路本地/保留地址、逐跳校验重定向。可经 `PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK=true` 放行内网。

> `llms_txt` 若 URL 非 llms.txt 清单(解析到 0 条)会直接报错并提示,而非"0 条成功"。

---

## 4. 增量同步 worker

`backend/rag/datasource/sync_worker.py` `run_sync()`,Celery 任务 `sync_datasource`(`backend/app/worker.py`)。

流程:`begin_sync`(**原子 claim**:`UPDATE ... WHERE status != syncing`,防并发重入;开 run、置 syncing)→ `discover` → 与 manifest diff(新增/删除免费,交集取正文比 `content_hash`)→ 分批 fetch 变更 → **upsert 走现有管道**(写 `file_store` `{kb_id}/docs/{datasource_key}/{path}` + 建/更 `KbFileEntity` + **提交后**入队 `enqueue_file_tasks.delay`)→ 删除走 `rag_service.delete_file` → `finalize_sync`(写 run 计数/report + 聚合态)。**零新增解析/嵌入代码。**

关键设计:
- **两阶段状态**:worker 段(抓取入库)结束→`ingesting`;Phase B(异步解析)由 `reconcile_document_statuses` **读时**从 `KbFileEntity.status` 派生 `synced/failed`(`sync-status` 端点触发)。
- **提交后入队**:enqueue 在 `session.commit()` 之后,否则 Celery worker(独立连接)读不到未提交的 KbFile → `File not found`(本次实现修复的核心竞态)。
- **re-sync 重入**:diff 仅当"内容未变 **且** `doc_status==synced`"才算 unchanged 跳过;`cancelled/failed` 文档即使 hash 未变也会被重新入库。
- **sphinx 部分发现保护**:`discovery_partial` 为真时**跳过删除**,避免临时抓取失败被误判为源端删除。
- **删除幂等**:`delete_file` 真失败时保留 manifest 行并标 failed(下次重试),不残留向量。

**取消 / 重解析**:
- `cancel_sync`:把未完成文件(pending/parsing/persisting)置 `cancelled`(运行中的解析任务经 `should_cancel_file_task` 协作停止),文档/数据源置 `cancelled`;之后「同步」会重新入库被取消的文档。
- `reparse_documents`:对已抓取入库的文件重置 `pending` + bump `file_version` 重新入队(恢复卡住/失败)。

**调度**:`datasource_sync_dispatch`(Celery beat,默认 300s,`PAIRAG_DATASOURCE_SYNC_DISPATCH_INTERVAL_SECONDS=0` 关闭)按各数据源 `sync_schedule`(间隔秒;装了 `croniter` 则支持 cron)/`next_sync_at` 跨租户派发到点的同步。`scheduler.py`。

---

## 5. Agent 检索工具(全知识库)

通用「全知识库」文件工具:`backend/tools/knowledgebase/knowledgebase_file_tools.py`,在 `agent_service.aget_tools()` 中**对每个 KB 无条件挂载**(不再按数据源 gating),与既有语义召回工具 `search-knowledgebase`(`aget_knowledgebase_tool`)并存。三个工具覆盖**整个知识库**(含手动上传的文件),不再局限于数据源文档。

> **历史**:早期为 `datasource_tool.py`,提供 search/catalog/keyword/fetch 四个**仅数据源**、按 `count_by_kb>0` gating 的工具。现已重构:语义 `search` 改由既有 `search-knowledgebase` 承担(去重,故此处不再单列),其余三个改为全 KB 语义并迁移至 `knowledgebase_file_tools.py`;`keyword` 更名为 `grep`。

| 工具 | 名称 | 数据来源 | 说明 |
|---|---|---|---|
| **catalog** | `catalog-{kb[:8]}` | `list_files`(关系库) | 列出 KB 内**全部文件**(文件名/标题/来源),**不读正文**;`query` 对 file_name+title 做大小写不敏感**子串**匹配(非 fuzzy);`limit≤200` |
| **grep** | `grep-{kb[:8]}` | chunk 预筛 + file_store | **字面短语**精确匹配(非正则),返回**行号+上下文**;参数 `pattern/context(≤10)/limit(≤200)`;预筛用 `contains(pattern, autoescape=True)`(`%`/`_` 不当通配符);`MAX_SCAN_FILES=200` 上限,**仅覆盖已建索引(有 chunk)的文件** |
| **fetch** | `fetch-{kb[:8]}` | file_store(回退 chunk 重组) | 按 `file_id`/`doc_id` 取全文;**`max_chars` 默认且硬上限 6000**(传入更大值会被夹到上限)+ `offset` 分页,返回 `truncated/next_offset`,防爆上下文 |

> **截断**是防 Agent 上下文溢出的关键:fetch 封顶 6000 字符,Agent 用 `offset=next_offset` 翻页或改用 grep 精确定位。
>
> 注:数据源专属的 `catalog`/`keyword` HTTP 端点(§6)仍存在并保留 `product/section/lang`、`path_prefix/datasource` 等数据源维度,供前端召回测试等使用;Agent 工具层已不再暴露这些维度。

---

## 6. REST API

均在 `/v1/config/knowledgebases` 前缀下(前端经 `/api/...` 代理 → 后端 `/v1/...`,代理 handler 在 `frontend/app/api/config/knowledgebases/[kb_id]/...`)。

**数据源管理**(`backend/api/v1/config_apis/datasource.py`):
- `POST/GET/PUT/DELETE /{kb_id}/datasources[/{ds_id}]`
- `POST /{ds_id}/sync | cancel | reparse | enable | disable`
- `GET /{ds_id}/documents | document?doc_id= | sync-runs[/{run_id}] | sync-status`

**检索测试 / 工具**(KB 级):
- `GET /{kb_id}/catalog?query=&product=&section=&lang=&limit=`(数据源目录,rapidfuzz 排序;**前端召回测试的 catalog 页现改用下方 `/files`**,本端点保留供其它数据源维度查询)
- `GET /{kb_id}/keyword?pattern=&doc_id=&path_prefix=&datasource=&context=&limit=`
- `GET /{kb_id}/file-content?file_id=|doc_id=&max_chars=&offset=`(`backend/api/v1/config_apis/knowledgebase.py`)
- `GET /{kb_id}/files?...&source=`(全 KB 文件列表;按来源筛选 manual/<datasource_key>;query 同时匹配 file_name + title。Agent `catalog` 工具与前端召回测试 catalog 页均用此端点)
- `POST /v1/retrieval`(既有召回 API,search 测试用)

---

## 7. 前端

- **知识库详情页新增「数据源」tab**(`frontend/app/knowledgebases/[kbId]/data-sources/data-sources-panel.tsx`):数据源卡片列表(状态徽章、进度条轮询、同步/取消/启停/编辑/删除、文档抽屉)、新增/编辑弹窗(类型动态表单 + 定时同步)。轮询仅对 `syncing/ingesting` 的源,3s,终态自停。
- **「检索测试」改名「召回测试」**,页内 5 个子 tab(`[kbId]/page.tsx`):
  - **召回**(原检索)、**目录**(catalog)、**关键词**(keyword)、**查看文件**(fetch,doc_id/file_id 取全文 + 加载更多分页)、**API 说明**(召回/目录/关键词/取文件的 curl + 请求/响应示例,自动填入当前 kb_id/tenant)。
  - 召回/目录/关键词结果均带「查看全文」直达 fetch。
- **文件列表统一展示**:主 KB 文件列表显示 `file_metadata.title`(回退 file_name)+ 命名空间路径副行;新增「来源」徽章列(数据源 key / 手动上传)+ 按来源筛选。
- i18n 在 `frontend/lib/translations.ts`(zh/en),`datasource` 与 `knowledgebase` 命名空间。

---

## 8. 命名 / 标识设计

| 维度 | 手动上传 | 数据源文档 |
|---|---|---|
| `file_name` | 原始文件名 | `{datasource_key}/{path}`(命名空间化,库内唯一、可溯源) |
| `file_metadata.title` | 文件名 / `.md` 首标题 | 适配器标题 / 正文首标题 |
| `datasource_key` | **无**(→「手动上传」) | 有(库内唯一,来源徽章/过滤的依据) |
| `doc_id`(规范键) | — | `{datasource_key}/{path}`,`VARCHAR(512)` |
| `file_id` | uuid | uuid |

多源同名(如 `index.md`)**不冲突**:store 路径、KbFile 唯一键(含 `message_id=ds-{datasource_id}`)、doc_id 三层均按数据源命名空间隔离。`datasource_key` 仅要求**知识库内唯一**(数据源归属单库,无需全局唯一)。

---

## 9. 存储兼容性

- **catalog / keyword / fetch**:只用**关系库(SQLite/MySQL/PG)+ file_store**,**不碰向量库**;仅用可移植 SQL(`== / != / LIKE / IN / order / limit / distinct`),rapidfuzz 在 Python 排序 → 三种关系库通用,且与任意向量后端(含 Elasticsearch)无关。
- **search**:走既有 `aquery` + `MetadataFilteringCondition`,兼容性 = 现有 KB 召回;`datasource_key/product/section/lang` 是普通 chunk 元数据键,过滤行为同既有 metadata 条件(ES/OpenSearch 动态映射默认 keyword 即可)。
- 注意:keyword 的 `LIKE` 预筛大小写在 PG(敏感)与 SQLite/MySQL(默认不敏感)不同,但最终匹配是 Python 区分大小写的 `pattern in line`,结果一致(差异仅多扫几个文件)。`file_metadata["title"].as_string()`(文件列表 title/source 过滤)经 SQLAlchemy 通用 JSON 适配三库。

---

## 10. 关键修复(本次实现中发现并解决)

| 问题 | 修复 |
|---|---|
| 同步后文件卡 `pending`、任务 `File not found` | enqueue 改到 `commit` 之后(竞态) |
| 同一数据源并发同步互相覆盖 | `begin_sync` 原子 claim |
| 用户 URL 造成 SSRF | `url_guard` 协议/私网/重定向校验 |
| sphinx 临时失败误删正常内容 | `discovery_partial` 跳过删除 |
| 删除失败仍移除 manifest → 残留向量 | 失败保留行 + 标记,下次重试 |
| MySQL 无法对 TEXT 建唯一索引 | `doc_id` 改 `VARCHAR(512)` |
| API 接受未实现的 local/github | 创建时按 registry 校验 |
| `sync_schedule` 无法置 null | `update` 用 `model_fields_set` |
| 前端弹窗/抽屉关闭后整页冻结 | Dialog/Sheet 受控常驻 + DropdownMenu `modal={false}` |
| fetch 全文撑爆 Agent 上下文 | 默认 6000 字符截断 + offset 分页 |

---

## 11. 配置 / 常量

- env:`PAIRAG_DATASOURCE_SYNC_DISPATCH_INTERVAL_SECONDS`(默认 300,0=关)、`PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK`(默认 false)。
- 常量:fetch `DEFAULT_FETCH_MAX_CHARS=6000`、keyword `MAX_SCAN_FILES=200`、catalog 候选 `CAP=5000`、同步 `batch_size=50` / `fetch_workers=6`、前端 fetch 窗口 `20000`。

---

## 12. 已知限制 / 后续

- keyword:仅**短语连续匹配**(非正则、不分词、不跨行、区分大小写)——刻意保持"精确"语义。
- catalog 上万文档:rapidfuzz 仍可用;>1w 建议切 `pg_trgm`/FTS5(spec §5.3)。
- `local` / `github` 适配器未实现(枚举预留)。
- 并发:`sync_datasource` 长任务占 Celery 槽,建议解析并发 `-c≥4` 或独立队列;大规模同步可加 in-flight 背压。
- 旧数据:已入库的旧手动文件无 `title`(展示回退 file_name);旧数据源文件需重新同步才刷新富化 metadata。

---

## 13. 主要改动文件

后端:`db/models/knowledgebase/datasource.py`、`common/knowledgebase/types.py`、`service/knowledgebase/datasource_service.py`、`service/knowledgebase/rag_service.py`、`service/knowledgebase/file_service.py`、`rag/datasource/*`(schema/base_adapter/registry/http_util/url_guard/sync_worker/scheduler/adapters/{llms_txt,sphinx})、`rag/file_item_utils.py`、`app/worker.py`、`service/agent/agent_service.py`、`tools/knowledgebase/knowledgebase_file_tools.py`、`api/v1/config_apis/{datasource,knowledgebase}.py`、`api/v1/routers.py`、`service/injection.py`。

前端:`app/knowledgebases/[kbId]/data-sources/data-sources-panel.tsx`、`app/knowledgebases/[kbId]/page.tsx`、`app/api/config/knowledgebases/[kb_id]/{datasources,catalog,keyword,file-content}/...`、`lib/translations.ts`、`api/v1/routers.py`。
