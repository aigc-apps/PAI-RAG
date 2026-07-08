# Agent Knowledge Base Management PRD

- **Status**: Draft v0.1
- **Owner**: Agent Platform
- **Scope**: 用户侧知识库管理交互设计，覆盖网站、OSS 地址文件、本地上传文件的导入、同步、验证、Agent 绑定。
- **Out of scope**: 底层解析器实现、向量库选型、embedding/rerank 模型接入细节、企业 SSO/OIDC。

## 1. 背景

Agent 需要从用户或企业提供的可信资料中检索信息，包括产品文档站、内部手册、OSS 上的 PDF/Markdown/HTML 文件、用户上传文档等。

如果只通过纯对话管理知识库，用户很难确认：

- 当前导入了哪些内容。
- 哪些文件成功、哪些失败。
- 失败原因是什么。
- Agent 当前到底使用了哪些知识库。
- 检索结果是否准确，是否能引用来源。

因此知识库应作为可管理、可观测、可审计的长期资产存在。

## 2. 产品结论

推荐采用：

```text
知识库管理界面为主入口 + 对话式管理为快捷入口
```

管理界面负责确定性操作：

- 创建知识库。
- 添加数据源。
- 查看文件/页面状态。
- 手动同步与重试。
- 测试检索效果。
- 绑定给 Agent。
- 管理权限和配置。

对话入口负责高效操作：

- “把这个网站导入到产品文档知识库”
- “重新同步售后知识库”
- “查看失败文件”
- “把产品文档绑定给 PAI Agent”

对话执行后必须落到可视化任务或知识库详情页，不能只用聊天消息作为最终状态载体。

## 3. 设计参考

主流产品的共性：

- Dify Knowledge 创建流程包含上传文件、同步网页、配置分段、配置索引与检索、等待处理完成。
- OpenAI File Search / Vector Store 将文件导入视为异步处理过程，需要观察 processing status，并将 vector store 绑定给 assistant 或 thread 使用。

参考链接：

- Dify Knowledge: https://docs.dify.ai/en/cloud/use-dify/knowledge/create-knowledge/introduction
- OpenAI File Search: https://developers.openai.com/api/docs/assistants/tools/file-search

## 4. 目标

### 4.1 用户目标

- 用户能在 5 分钟内创建一个可用知识库。
- 用户能导入网站、OSS 路径、本地文件。
- 用户能清楚知道每个文件/页面是否已被 Agent 使用。
- 用户能看到失败原因并重试。
- 用户能在绑定 Agent 前测试检索效果。
- Agent 回答时能展示引用来源，降低幻觉风险。

### 4.2 产品目标

- 将知识库从“黑盒导入”变成“可管理资产”。
- 降低非技术用户配置成本。
- 支持后续扩展到多数据源、多租户、定时同步、审计。
- 保证 Agent 只能检索被显式绑定的知识库。

### 4.3 非目标

MVP 不追求：

- 全站深度爬取复杂动态网页。
- OCR 和扫描件识别。
- 表格结构化问答。
- 文档级细粒度 ACL。
- 多版本索引回滚。
- 自由编辑 chunk 文本。
- 自动质量评估闭环。

## 5. 用户角色

### 5.1 Admin

管理员负责：

- 创建、删除知识库。
- 添加网站/OSS/上传文件数据源。
- 管理用户可见范围。
- 绑定知识库到 Agent。
- 查看导入任务和错误。
- 调整高级检索配置。

### 5.2 Knowledge Editor

知识库维护者负责：

- 上传或更新文件。
- 添加或编辑自己有权限的数据源。
- 手动同步和重试失败文件。
- 测试检索结果。

### 5.3 Agent User

普通使用者负责：

- 在聊天中使用已绑定知识库。
- 查看回答引用来源。
- 反馈“答案不准”或“没有检索到”。

普通用户不应默认拥有全局知识库配置权限。

## 6. 核心用户需求

### 6.1 创建知识库

用户故事：

```text
作为管理员，我想创建一个“产品文档”知识库，并快速导入第一批资料，
这样我可以让 Agent 基于这些资料回答问题。
```

需求：

- 输入名称、描述。
- 选择可见范围。
- 选择创建方式：
  - Empty
  - Upload files
  - Import website
  - Import OSS path
- 默认使用推荐配置。
- 创建完成后自动进入导入任务页或知识库详情页。

### 6.2 导入网站

用户故事：

```text
作为管理员，我想导入 https://docs.example.com，
但只导入 /docs/*，排除 /blog/*，
这样 Agent 不会检索无关内容。
```

需求：

- 支持单 URL。
- 支持 sitemap URL。
- 支持同域名 crawl。
- 支持 include/exclude path pattern。
- 支持最大页面数、最大深度。
- 显示导入范围预览。
- 显示抓取失败页面。
- 支持手动重新同步。

MVP 推荐优先级：

1. Single URL
2. Sitemap
3. Limited crawl

不建议 MVP 默认无限全站爬取。

### 6.3 导入 OSS

用户故事：

```text
作为用户，我想导入 oss://bucket/path/ 下的文件，
并且不在浏览器里填写 AK/SK。
```

需求：

- 输入 region、bucket、prefix。
- 选择文件类型过滤：
  - PDF
  - Markdown
  - HTML
  - TXT
  - DOCX
  - CSV
- 支持最大文件大小限制。
- 支持连接测试。
- 支持 object list 预览。
- 支持按 etag / last_modified 增量同步。
- 凭证走服务端配置或用户 RAM 授权，不在前端保存 AK/SK。

### 6.4 文件管理

用户故事：

```text
作为知识库维护者，我想看到所有导入文件的处理状态，
这样我知道哪些内容已经生效。
```

需求：

- 文档列表展示：
  - 标题 / 文件名 / URL / OSS key
  - 来源类型
  - 文件大小
  - 状态
  - chunk 数
  - 最近索引时间
  - 错误原因
- 支持操作：
  - Retry
  - Re-parse
  - Delete
  - Disable
  - View source

### 6.5 文档元数据管理

用户故事：

```text
作为知识库维护者，我想按来源、标签、文件类型、状态筛选文档，
并编辑标题、标签、分类等业务元数据，
这样我可以维护一个可查、可治理的知识库。
```

需求：

- 支持按元数据查询：
  - Source type
  - Source URI / URL / OSS path
  - File type
  - Status
  - Tags
  - Category
  - Created by
  - Indexed time
  - Error code
- 支持展示系统元数据：
  - `doc_id`
  - `source_id`
  - `source_uri`
  - `hash`
  - `etag`
  - `chunk_count`
  - `indexed_at`
- 支持编辑用户侧元数据：
  - Title
  - Description
  - Tags
  - Category
  - Visibility
  - Custom metadata
- 不允许编辑系统元数据：
  - `doc_id`
  - `source_uri`
  - `hash`
  - `etag`
  - `chunk_count`
  - `embedding_ref`

设计原则：

- 系统元数据用于同步、去重、引用、增量索引，必须由系统维护。
- 用户元数据用于检索过滤、治理和展示，可以编辑。
- 所有元数据修改需要记录修改人和修改时间。

### 6.6 Chunk 管理

用户故事：

```text
作为知识库维护者，我想查看某个文档被切成了哪些 chunk，
并在切分不合理时重新解析或禁用错误 chunk，
这样我可以提高 Agent 的检索质量。
```

需求：

- 支持查看文档 chunks。
- 支持按 chunk 内容搜索。
- 展示每个 chunk 的：
  - Chunk index
  - Heading path
  - Text preview
  - Token count
  - Status
  - Metadata
  - Last embedded time
- 支持操作：
  - View full chunk
  - Disable chunk
  - Re-enable chunk
  - Rebuild chunks for document
  - Rebuild chunks for knowledge base
- 支持调整 parser/chunk 配置后重建索引。
- MVP 不支持自由编辑 chunk 文本。

不支持自由编辑 chunk 的原因：

- Chunk 是索引产物，不是源文档。
- 源文档重新同步后，手工编辑可能被覆盖。
- Chunk 与原文不一致会导致引用来源失真。
- 修改 chunk 需要重新 embedding 和重新索引，状态管理更复杂。

后续如果需要“纠错”，应采用 override/correction 模型：

```text
source document
  -> parsed chunks
  -> user correction / override
  -> effective indexed chunks
```

该模型必须支持审计、回滚和源文档更新冲突提示。

### 6.7 检索测试

用户故事：

```text
作为管理员，我想在绑定给 Agent 前测试“如何创建 EAS 服务”，
看看知识库是否能命中文档。
```

需求：

- 输入测试问题。
- 展示命中的 chunks。
- 展示 score、source、document、heading。
- 支持打开原文。
- 支持显示 Agent 实际会拿到的上下文。
- 支持调试参数：
  - top_k
  - score_threshold
  - retrieval mode

### 6.8 Agent 绑定

用户故事：

```text
作为管理员，我想把“产品文档”和“售后 FAQ”绑定给 PAI Agent，
这样该 Agent 只检索这些知识库。
```

需求：

- 一个 Agent 可绑定多个知识库。
- 一个知识库可绑定多个 Agent。
- 默认不自动绑定到所有 Agent。
- 每个绑定支持配置：
  - enabled
  - top_k
  - score_threshold
  - force citation
  - query rewrite
- Agent 设置页展示已绑定知识库。

## 7. 信息架构

新增一级入口：

```text
Knowledge
├── Knowledge List
├── Knowledge Detail
│   ├── Overview
│   ├── Sources
│   ├── Documents
│   ├── Metadata
│   ├── Chunks
│   ├── Retrieval Test
│   ├── Agent Binding
│   └── Settings
└── Import Job Detail
```

## 8. 页面规格

### 8.1 Knowledge List

目标：

- 让用户快速知道有哪些知识库、是否可用、是否有错误。

字段：

| 字段 | 说明 |
|------|------|
| Name | 知识库名称 |
| Description | 简要描述 |
| Status | Empty / Processing / Ready / Has Errors |
| Documents | 文档数量 |
| Chunks | chunk 数 |
| Sources | 数据源数量 |
| Agents | 已绑定 Agent 数 |
| Last Sync | 最近同步时间 |

操作：

- Create Knowledge Base
- Open
- Sync
- Test
- Bind Agent

### 8.2 Create Knowledge Base

采用向导式流程。

#### Step 1: Basic Info

字段：

- Name
- Description
- Visibility
- Owner

#### Step 2: Import Source

选项：

- Empty
- Upload files
- Website
- OSS

#### Step 3: Processing Settings

默认折叠高级配置。

默认文案：

```text
Recommended settings work for most product documents. You can adjust them later.
```

高级配置：

- Parser mode
- Chunk size
- Chunk overlap
- Embedding profile
- Vector store profile
- Retrieval mode
- Rerank

交互规则：

- 默认不要求用户理解 embedding 和 vector store；系统使用管理员配置的默认 profile。
- 管理员可展开高级配置，选择已配置的 embedding profile 和 vector store profile。
- 普通用户只能看到“索引配置：推荐配置”或“由管理员配置”，不能看到 endpoint、secret、AK/SK。
- 如果修改 chunk 或 embedding 相关配置，需要提示“将触发重建索引，完成前继续使用当前可用索引”。
- 如果 profile 未配置或连接测试失败，创建按钮禁用，并提示管理员进入 Settings 修复。

用户需要填写：

| 角色 | 必填项 | 可选项 | 不需要填写 |
|------|--------|--------|------------|
| 普通用户 | Name、Description、Import Source | Tags、Visibility | Embedding key、Vector endpoint、Index name |
| Editor | Name、Source、Parser/Retrieval 参数 | Embedding/Vector profile | 明文密钥 |
| Admin | Provider secret 引用、默认 profile、权限策略 | Rerank、Keyword index | 用户源文档内容 |

#### Step 4: Confirm

展示：

- 来源配置摘要。
- 预计导入数量。
- 文件类型限制。
- 是否立即开始导入。

点击 `Start Import` 后进入 Import Job Detail。

### 8.3 Overview

展示：

- Status card
- Documents by status
- Last sync
- Latest import jobs
- Top errors
- Bound agents

快捷操作：

- Add Source
- Sync Now
- Test Retrieval
- Bind Agent

### 8.4 Sources

字段：

| 字段 | 说明 |
|------|------|
| Type | Upload / Website / OSS |
| Summary | URL / bucket prefix / file batch |
| Sync Mode | Manual / Scheduled |
| Status | Ready / Paused / Error |
| Last Sync | 最近同步 |
| Error | 最近错误 |

操作：

- Sync
- Edit
- Pause
- Delete

#### Website Source Form

字段：

- Start URL
- Import mode:
  - Single URL
  - Sitemap
  - Crawl within domain
- Include patterns
- Exclude patterns
- Max pages
- Max depth
- Respect robots.txt
- Schedule

#### OSS Source Form

字段：

- Region
- Bucket
- Prefix
- File types
- Max file size
- Credential mode:
  - Service role
  - User RAM authorization
- Sync mode:
  - Manual
  - Scheduled

操作：

- Test connection
- Preview objects
- Save source

### 8.5 Documents

字段：

| 字段 | 说明 |
|------|------|
| Name | 标题 / 文件名 |
| URI | URL / OSS key / uploaded file |
| Source | 数据源 |
| Status | 当前处理状态 |
| Size | 文件大小 |
| Tags | 用户标签 |
| Category | 用户分类 |
| Chunks | chunk 数 |
| Indexed At | 最近索引时间 |
| Error | 错误原因 |

状态：

```text
Queued
Fetching
Parsing
Chunking
Embedding
Indexed
Failed
Disabled
Deleted
```

操作：

- View
- Edit metadata
- Retry
- Re-parse
- Disable
- Delete

失败原因枚举：

- `url_unreachable`
- `permission_denied`
- `file_too_large`
- `unsupported_file_type`
- `parse_failed`
- `empty_content`
- `embedding_failed`
- `index_failed`

#### Document Detail Drawer

打开文档后展示：

- Basic info
- Source info
- System metadata
- User metadata
- Chunk preview
- Latest ingestion logs

可编辑字段：

- Title
- Description
- Tags
- Category
- Visibility
- Custom metadata

只读字段：

- Document ID
- Source ID
- Source URI
- Content hash
- ETag
- Last modified
- Chunk count
- Indexed at

### 8.6 Metadata

目标：

- 面向知识库治理和调试，提供跨文档的元数据查询与批量编辑能力。

筛选条件：

- Source type
- Source
- File type
- Status
- Tags
- Category
- Created by
- Indexed time
- Error code

支持操作：

- Batch add/remove tags
- Batch set category
- Batch disable documents
- Export metadata CSV

限制：

- 不允许批量修改系统元数据。
- 批量操作必须展示影响数量并二次确认。

### 8.7 Chunks

目标：

- 让用户观察和治理切分结果，但不把 chunk 当作长期知识正文维护。

字段：

| 字段 | 说明 |
|------|------|
| Chunk | chunk 序号 |
| Document | 所属文档 |
| Heading | 标题路径 |
| Text Preview | 内容预览 |
| Tokens | token 数 |
| Status | active / disabled |
| Indexed At | 最近 embedding 时间 |

操作：

- View full chunk
- Disable chunk
- Re-enable chunk
- Rebuild document chunks

Chunk detail 展示：

- Full text
- Metadata
- Source document
- Heading path
- Character offset
- Embedding status

MVP 明确不提供：

- 直接编辑 chunk text。
- 手工创建新 chunk。
- 将 chunk 脱离源文档独立维护。

如果后续支持 correction/override，需要新增：

- Correction author
- Correction reason
- Effective text
- Original text
- Conflict status
- Revert action

### 8.8 Retrieval Test

输入：

- Query
- Optional top_k
- Optional score threshold

输出：

- User-facing preview
- Debug results

Debug result 字段：

| 字段 | 说明 |
|------|------|
| Score | 检索分数 |
| Text | chunk 片段 |
| Document | 文档标题 |
| Source | URL / OSS path |
| Heading | 标题层级 |
| Chunk ID | 调试用 |

用户提示：

```text
If the expected document does not appear here, the Agent probably will not use it either.
```

### 8.9 Agent Binding

字段：

| 字段 | 说明 |
|------|------|
| Agent | Agent 名称 |
| Enabled | 是否启用 |
| Top K | 最大召回片段数 |
| Threshold | 分数阈值 |
| Force Citation | 是否强制引用来源 |

操作：

- Bind Agent
- Edit binding
- Disable binding
- Remove binding

### 8.10 Settings

配置：

- Name
- Description
- Visibility
- Indexing & Retrieval
  - Embedding profile
  - Vector store profile
  - Keyword index profile
  - Rerank profile
  - Chunk size / overlap
  - Retrieval mode / top_k / threshold
- Default parser settings
- Default retrieval settings
- Default metadata schema
- Delete knowledge base
- Rebuild index

`Indexing & Retrieval` 权限：

- Admin 可以配置全局 provider 和知识库默认 profile。
- Editor 可以在 admin 允许的 profile 中选择，并触发重建。
- 普通用户只能查看知识库是否可查询、当前检索模式和最近索引时间。

连接配置展示规则：

- Secret 只展示 `configured` / `missing`，不展示值。
- Vector store endpoint 默认只对 admin 展示。
- Index name / namespace 默认只对 admin/editor 展示。
- 所有连接测试由后端执行，前端不接收 provider 密钥。

危险操作必须二次确认：

- Delete knowledge base
- Rebuild all documents
- Rebuild all chunks
- Remove source and documents

## 9. 对话式管理

对话管理是快捷入口，不是唯一入口。

支持语义：

```text
把 https://docs.example.com 导入到“产品文档”
重新同步“售后知识库”
查看“产品文档”失败的文件
给“产品文档”里所有 EAS 文档加上标签 pai-eas
查看“计费说明.pdf”的 chunk 切分结果
禁用“计费说明.pdf”里第 12 个 chunk
重新解析“计费说明.pdf”
把“产品文档”绑定给 PAI Agent
测试“产品文档”：如何创建 EAS 服务？
```

规则：

- 删除、全量重建、批量禁用、批量元数据修改必须确认。
- 操作完成后返回任务卡片。
- 任务卡片必须包含：
  - Job status
  - Success count
  - Failed count
  - Link to job detail
  - Link to knowledge detail
- 对话中不得隐藏失败状态。

## 10. Agent 使用体验

Agent 回答时，如果使用知识库，应该展示来源。

推荐回答结构：

```text
Answer...

Sources:
1. Document title - source URL
2. OSS object path - document name
```

如果没有命中，应允许 Agent 说明：

```text
I could not find relevant content in the configured knowledge bases.
```

不要让 Agent 声称“根据知识库”但没有引用来源。

## 11. 权限设计

MVP 权限：

| 操作 | Admin | Editor | User |
|------|:-----:|:------:|:----:|
| Create KB | ✓ | Optional | - |
| Delete KB | ✓ | - | - |
| Add source | ✓ | ✓ | - |
| Upload file | ✓ | ✓ | Optional |
| Sync source | ✓ | ✓ | - |
| Edit document metadata | ✓ | ✓ | - |
| View chunks | ✓ | ✓ | View-only |
| Disable chunks | ✓ | ✓ | - |
| Rebuild chunks | ✓ | ✓ | - |
| Bind Agent | ✓ | - | - |
| Test retrieval | ✓ | ✓ | View-only |
| Chat retrieval | ✓ | ✓ | ✓ |

隔离原则：

- Agent 只能使用显式绑定的知识库。
- 用户只能访问自己有权限的知识库。
- OSS 凭证不进入前端。
- 需要记录谁添加了数据源、谁触发了同步、谁绑定了 Agent。

## 12. 数据模型建议

### 12.1 KnowledgeBase

```text
id
name
description
owner_user_id
visibility
status
embedding_config
vector_store_config
keyword_index_config
rerank_config
active_index_version_id
document_count
chunk_count
created_at
updated_at
```

### 12.2 DataSource

```text
id
kb_id
type              # upload | website | oss
config            # JSON
sync_mode         # manual | scheduled
status
last_sync_at
created_by
created_at
updated_at
```

### 12.3 Document

```text
id
kb_id
source_id
uri
title
description
mime_type
size_bytes
hash
etag
last_modified
tags
category
visibility
custom_metadata
status
chunk_count
error_code
error_message
indexed_at
created_at
updated_at
```

### 12.4 IngestionJob

```text
id
kb_id
source_id
triggered_by
trigger_type      # manual | schedule | conversation
status
total_count
succeeded_count
failed_count
started_at
finished_at
error_summary
```

### 12.5 Chunk

```text
id
document_id
kb_id
chunk_index
text
metadata
token_count
embedding_ref
status              # active | disabled
heading_path
char_start
char_end
disabled_by
disabled_reason
created_at
updated_at
```

### 12.6 AgentKnowledgeBinding

```text
id
agent_id
kb_id
enabled
top_k
score_threshold
force_citation
query_rewrite_enabled
created_at
updated_at
```

## 13. 状态机

### 13.1 KnowledgeBase Status

```text
empty
processing
ready
has_errors
disabled
```

计算规则：

- `empty`: 无 indexed 文档。
- `processing`: 存在 running job 或 processing document。
- `ready`: 至少一个 indexed 文档，且无 failed 文档。
- `has_errors`: 至少一个 indexed 文档，同时存在 failed 文档。
- `disabled`: 人工停用。

### 13.2 Document Status

```text
queued
fetching
parsing
chunking
embedding
indexed
failed
disabled
deleted
```

### 13.3 IngestionJob Status

```text
queued
running
completed
completed_with_errors
failed
cancelled
```

### 13.4 Chunk Status

```text
active
disabled
stale
deleted
```

说明：

- `active`: 可被检索。
- `disabled`: 用户或系统禁用，不进入检索结果。
- `stale`: 源文档已变化，等待重新解析或重新 embedding。
- `deleted`: 所属文档删除后软删除。

## 14. MVP 范围

MVP 必须做：

- Knowledge list。
- Create knowledge base。
- Upload files。
- Website single URL + sitemap。
- OSS prefix import。
- Documents list + status。
- Metadata query and document-level metadata edit。
- Chunk view。
- Rebuild chunks for a document。
- Manual sync。
- Retry failed document。
- Retrieval test。
- Agent binding。
- Basic permissions。

MVP 暂不做：

- OCR。
- 动态网页浏览器渲染。
- 自动定时同步。
- 多版本索引回滚。
- 文档级 ACL。
- 自由编辑 chunk 文本。
- 手工创建 chunk。
- Chunk correction / override。
- 自动评测。
- 高级表格解析。

## 15. 关键验收标准

### 15.1 创建与导入

- 用户能创建空知识库。
- 用户能上传文件并看到 processing -> indexed。
- 用户能输入网站 URL 并导入页面。
- 用户能输入 OSS bucket/prefix 并预览 object。
- 导入失败时必须展示可理解错误。

### 15.2 检索测试

- 用户能输入 query 并看到命中 chunk。
- 命中结果包含 source URL 或 OSS path。
- 用户能调整 top_k 和 threshold。

### 15.3 元数据管理

- 用户能按来源、状态、文件类型、标签筛选文档。
- 用户能编辑 title、description、tags、category、visibility。
- 用户不能编辑 doc_id、source_uri、hash、etag、chunk_count 等系统字段。
- 元数据修改记录修改人和修改时间。

### 15.4 Chunk 管理

- 用户能查看某个文档的 chunk 列表。
- 用户能查看 chunk 全文、heading path、metadata、token count。
- 用户能禁用和恢复 chunk。
- 被禁用 chunk 不会出现在检索结果中。
- 用户能触发单文档 chunk 重建。
- MVP 不提供直接编辑 chunk text。

### 15.5 Agent 绑定

- 用户能把知识库绑定给 Agent。
- 未绑定知识库不会被 Agent 检索。
- Agent 回答能展示引用来源。

### 15.6 权限

- 普通用户不能修改全局知识库配置。
- 非授权用户不能看到无权限知识库。
- OSS AK/SK 不出现在前端、日志或响应中。

### 15.7 可观测性

- 每次导入有 job 记录。
- 每个 job 展示成功/失败数量。
- 每个失败文档有错误码和错误消息。

## 16. 分阶段计划

### Phase 1: 可用闭环

- Knowledge list/detail。
- Upload file。
- Website single URL/sitemap。
- OSS prefix。
- Manual sync。
- Documents status。
- Metadata filters and document metadata edit。
- Chunk view and single-document rebuild。
- Retrieval test。
- Agent binding。

### Phase 2: 可靠性增强

- 定时同步。
- 批量重试。
- 错误报告下载。
- 增量同步优化。
- 来源变更检测。
- Batch metadata editing。
- Chunk disable/re-enable。
- 引用质量反馈。

### Phase 3: 企业能力

- 文档级 ACL。
- 审计日志。
- SSO/OIDC 集成。
- 多 workspace。
- 版本回滚。
- Chunk correction / override with audit and rollback。
- 自动检索质量评估。

## 17. Open Questions

- OSS 导入是否统一走服务端角色，还是复用用户 Aliyun RAM 授权？
- 网站导入是否必须遵守 robots.txt，还是允许管理员配置覆盖？
- 是否允许普通用户创建个人知识库？
- Agent 绑定是在 Agent Settings 下管理，还是只在 Knowledge 下管理，或两边都能管理？
- 文件删除是否软删除，是否保留历史索引版本用于回滚？
- 是否需要支持知识库 Marketplace 或模板？
- 文档元数据是否需要自定义 schema，还是先使用自由 JSON metadata？
- Chunk 禁用是否作为 MVP 必做，还是先只做 view + rebuild？
- 如果源文档更新后用户曾做过 chunk correction，冲突应该自动失效还是进入人工 review？
- 元数据批量编辑是否需要审批流？

## 18. 推荐默认值

```yaml
chunk:
  size: 1000
  overlap: 150
retrieval:
  mode: hybrid
  top_k: 6
  score_threshold: 0.25
  rerank: false
website:
  max_pages: 200
  max_depth: 3
  respect_robots_txt: true
oss:
  max_file_size_mb: 50
  allowed_extensions:
    - .pdf
    - .md
    - .txt
    - .html
    - .docx
```

## 19. 用户文案建议

导入中：

```text
We are parsing and indexing your documents. The Agent will only use documents after they are indexed.
```

部分失败：

```text
Some documents failed to import. Indexed documents are already available, and failed documents can be retried.
```

检索测试无结果：

```text
No relevant chunks were found. Try lowering the threshold, adding more documents, or checking whether the source has been indexed.
```

删除知识库：

```text
This will remove the knowledge base from all Agents and stop future retrieval. Existing chat history will not be changed.
```

## 20. 最终建议

知识库管理必须有界面。纯对话适合快速触发操作，但不适合作为唯一控制面。

第一版应优先打通：

```text
创建知识库 -> 导入来源 -> 查看状态 -> 检索测试 -> 绑定 Agent -> 回答引用来源
```

只要这条闭环稳定，后续再扩展定时同步、复杂解析、精细权限和质量评估。
