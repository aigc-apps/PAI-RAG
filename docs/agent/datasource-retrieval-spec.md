# 知识库 DataSource 检索协议 (KB DataSource Retrieval Spec)

- **状态**: Draft v0.2
- **范围**: 定义外部文档数据源(datasource)接入 RAG 知识库后的完整契约——**统一 metadata schema** + **数据源适配器** + **增量同步** + **4 个检索工具**(`search` / `keyword` / `catalog` / `fetch`)。
- **目标**: 任意 datasource(aliyun llms.txt 站、readthedocs/Sphinx 站…)经各自适配器抽取为统一 schema 后,Agent 用同一套工具检索,数据源对 Agent 透明。
- **当前数据源**: ① 阿里云帮助文档(llms.txt) ② readthedocs/Sphinx 站点。

---

## 1. 设计原则

1. **四轴正交**:检索能力分四条互不重叠的轴,每轴一个工具。
   | 轴 | 解决 | 工具 |
   |----|------|------|
   | 语义 | 模糊意图("讲 XX 的内容") | `search` |
   | 精确/关键词 | 确切 token(错误码 `137`、配置键 `eventTime`) | `keyword` |
   | 元数据/目录 | 找文件、按结构过滤、浏览 | `catalog` |
   | 取全文 | 切片太碎,读整篇/上下文 | `fetch` |
2. **源无关**:适配器把异构源归一为统一 schema;下游切片/索引/4 工具与源类型解耦。
3. **metadata 驱动**:`fetch` 寻址、`search` 过滤、`catalog` 匹配、增量同步,全部依赖同一套 metadata。
4. **来源可溯**:每个返回项必带 `source_url`。
5. **工具克制 + 优雅降级**:只暴露 4 个工具;某路(向量/rerank)不可用时自动退化并标注 `degraded`。

---

## 2. Metadata Schema

适配器抽取文档时产出两级元数据。

### 2.1 Document

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `doc_id` | string | ✓ | 全局稳定唯一 ID,`"{datasource}/{path}"`。重抓后保持稳定 |
| `datasource` | string | ✓ | 数据源标识,如 `aliyun_docs` / `easyrec_docs` |
| `path` | string | ✓ | datasource 内相对路径,如 `zh/pai/billing-of-eas.md` |
| `title` | string | ✓ | 文档标题 |
| `source_url` | string | ✓ | 原文网页 URL |
| `source_site` | string | ✓ | 源站域名 |
| `product` | string | – | 所属产品/集合 |
| `section` | string | – | 章节分组 |
| `summary` | string | – | 一句话摘要 |
| `lang` | string | – | `zh` / `en` |
| `content_hash` | string | ✓ | 正文 sha256,增量同步与变更追踪的依据 |
| `fetched_at` | date | ✓ | 抓取日期 (ISO 8601) |
| `fetched_from` | string | ✓ | 抽取方式,如 `aliyun-llms.txt` / `sphinx-html` |
| `tags` | string[] | – | 自由标签 |

### 2.2 Chunk

| 字段 | 类型 | 必填 | 说明 |
|------|------|:--:|------|
| `chunk_id` | string | ✓ | `"{doc_id}#{chunk_index}"` |
| `doc_id` | string | ✓ | 所属文档 |
| `chunk_index` | int | ✓ | 文档内序号(0-based),用于邻域扩展 |
| `heading_path` | string[] | – | 切片所在标题层级,如 `["计费","EAS 计费说明"]` |
| `text` | string | ✓ | 切片正文 |
| `char_start`/`char_end` | int | – | 原文偏移,用于精确定位 |

**约定**:`search` 返回的 chunk 内嵌其 Document 的全部 metadata,Agent 一次拿全寻址键 + `source_url`。

---

## 3. 数据源适配器 (Adapters)

每个数据源由一个 **adapter** 抽取为统一 schema。adapter 是源类型与统一协议之间唯一的耦合点;新增源 = 新增 adapter,下游零改动。

### 3.1 Adapter 契约

| 阶段 | 职责 |
|------|------|
| **discover** | 列出当前全量文档清单 → `[{path, title, section, summary?, source_url}]` |
| **fetch** | 取单篇正文 → 标准 Markdown |
| **emit** | 产出 §2.1 Document metadata(必含 `doc_id`/`content_hash`/`source_url`)+ 落盘 |

约束:正文统一为 Markdown;正文内相对链接/图片重写为**源站绝对 URL**;`content_hash` = 正文(不含 frontmatter)的 sha256。

### 3.2 已支持:`llms.txt` 适配器(阿里云帮助文档)

- **适用**:提供官方 `llms.txt` 的站点(`help.aliyun.com` 及其子产品)。
- **discover**:拉 `https://help.aliyun.com/zh/<product>/llms.txt` → 解析 `- [标题](url.md): 摘要` → 全量清单 + 章节(`## 分节`) + 现成 `summary`。子产品给完整 `--llms-url`。
- **fetch**:直接下载官方 `.md` 源(高质量,无需 HTML 转换)。
- **落盘**:按 URL path,`zh/<product>/...`。
- **现状**:PAI 2140 + PAI-Rec 111 = **2251 篇**。
- **优势**:清单权威 → 新增/删除可仅靠清单 diff **免费检测**;`summary` 官方现成。
- **限制**:个别 `.md` 端点失效需网页兜底;表格是内嵌 HTML(保留,不转)。

### 3.3 已支持:`sphinx` 适配器(readthedocs / Sphinx 站)

- **适用**:`sphinx_rtd_theme` 类 readthedocs 站(EasyRec、TorchEasyRec 等同构站)。
- **discover**:**无 llms.txt** → 解析首页侧边栏 `toctree` 取种子页 + 章节(caption)→ **BFS** 跟随每页正文内指向同站(base 前缀下)的 `.html` 链接,递归发现深层子页(解决 toctree 只列 l1 漏子页的问题)。
- **fetch**:抓 HTML → 提取正文区 `div[itemprop="articleBody"]` → `markdownify` 转 md;非标准页(如自定义生成的 protobuf 参考)fallback 到整个 `<body>` 去导航/页脚后转。
- **落盘**:按 URL path,`<lang>/<version>/...`。换站只需改 `--base-url`。
- **现状**:EasyRec **112 篇** + TorchEasyRec **55 篇**。
- **优势**:通用,同构站零改动复用。
- **限制**:无官方清单 → 靠 BFS(需排除 `_downloads`/`_static`/`_modules` 等);HTML→md 有损;无官方 `summary`(用 toctree 文本/h1 代替)。

### 3.4 新增数据源

实现 §3.1 三阶段即可。常见类型与发现方式:

| 源类型 | discover | fetch |
|--------|----------|-------|
| `llms.txt` | 解析 llms.txt 清单 | 下载阿里云官方 `.md` |
| `sphinx` | toctree + BFS | HTML 正文区 → md |
| `local` | 本地上传文件 | 上传 |
| `github-repo` | git 列 `docs/**` | raw md / rst→md |


---

## 4. 增量同步

适配器每次运行 = 一次增量同步;只对变化的文档重建切片/向量,不全量重跑。

### 4.1 同步流程

```
1. discover            → 当前全量 doc_id 集合(+ 每篇 source_url/section/summary)
2. 与上次 manifest 对比(doc_id → content_hash):
     新增 = 当前有、上次无
     删除 = 上次有、当前无
     更新 = doc_id 相同、content_hash 变
     不变 = 跳过
3. 仅对 新增+更新:fetch 正文 → 算 content_hash → 写盘
4. 输出变更集给 RAG 服务:
     upsert(新增/更新的 doc + 其重切片)
     delete(删除的 doc_id 及其 chunks/向量)
5. 写新 manifest + 变更报告(+N 新增 / ~M 更新 / -K 删除)
```

### 4.2 变更检测成本(按源类型)

源站普遍**不返回 `ETag`/`Last-Modified`**(实测 aliyun `.md` 与页面均无),故"更新"检测无法靠 HTTP 304 省流量,需下正文比 `content_hash`。但"新增/删除"在有清单的源上免费:

| 源类型 | 新增 / 删除 | 更新 |
|--------|------------|------|
| `llms.txt` | 清单 diff,**免费**(不下正文) | 下 `.md` 比 `content_hash` |
| `sphinx` | 重跑 BFS 得清单 | 抓 HTML 比 `content_hash` |

> 优化:`llms.txt` 的 `summary` 变化可作"高概率更新"信号优先处理;但小改不动摘要会漏,要不漏仍需 `content_hash` 兜底。

### 4.3 manifest

每个 datasource 维护一份 `manifest`(`doc_id → {content_hash, source_url, title, section, fetched_at}`):

- 既是**同步状态**(下次 diff 的基准),
- 也是 **`catalog` 工具的数据来源**(文件名/标题模糊匹配 + 结构化过滤都在 manifest 上跑,无需读正文)。

### 4.4 RAG 侧接口

adapter 产出变更集,RAG 服务暴露两个写接口即可支撑增量:

```jsonc
upsert(documents: [{ ...§2.1 metadata, content }])   // 新增/更新 → 重切片 + 重嵌入 + 重建索引
delete(doc_ids: string[])                            // 删除 → 清 chunks/向量/catalog
```

---

## 5. 检索工具规范

通用返回包裹:

```jsonc
{ "ok": true, "degraded": null, "results": [ /* ... */ ], "total": 12 }
```

错误:`{ "ok": false, "error": "<code>", "message": "..." }`。空命中:`ok:true, results:[]`。

### 5.1 `search` — 语义 + 关键词混合召回 + 重排

模糊意图首选。多路召回(BM25 + 向量)融合或 rerank,返回 top-k 切片。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | string | — | 自然语言查询(必填) |
| `k` | int | 8 | 返回切片数 |
| `filters` | object | – | `{ datasource?, product?, section?, lang? }`,先限定再搜 |
| `mode` | enum | `hybrid` | `hybrid`/`vector`/`keyword`(调试用) |

```jsonc
// results[]
{
  "chunk_id": "easyrec_docs/en/latest/proto.md#3",
  "doc_id":   "easyrec_docs/en/latest/proto.md",
  "score":    0.87,                         // query 内相对分，跨 query 不可比
  "text":     "...切片正文...",
  "heading_path": ["DATA & FEATURE", "Protocol Documentation"],
  "metadata": { "title":"Protocol Documentation", "product":"EasyRec",
                "source_url":"https://easyrec.readthedocs.io/en/latest/proto.html",
                "datasource":"easyrec_docs" }
}
```

### 5.2 `keyword` — 精确串 / 正则定位

确切标识符首选(向量召回不了 `137`/`gu8xf`/函数名)。字面或正则匹配,可限定范围。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pattern` | string | — | 字面串或正则(必填) |
| `is_regex` | bool | false | 是否按正则 |
| `scope` | object | – | `{ doc_id?, datasource?, path_prefix? }` |
| `limit` | int | 20 | 最多匹配数 |
| `context` | int | 2 | 每个匹配返回前后 N 行 |

```jsonc
// results[]
{ "doc_id":"aliyun_docs/zh/pai/user-guide/dlc-faq.md", "line":42,
  "match":"遇到错误码137怎么办", "context":"...前后 N 行...",
  "source_url":"https://help.aliyun.com/zh/pai/user-guide/dlc-faq" }
```

### 5.3 `catalog` — 元数据层:找文件 + 过滤 + 浏览

在 manifest(元数据,非内容)上检索。三合一:① 文件名/标题模糊匹配;② 结构化过滤;③ 列举/浏览。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | string | – | 对 `title`+`path`(+`summary`)模糊匹配;省略=纯列举 |
| `filters` | object | – | `{ datasource?, product?, section?, lang? }` |
| `limit` | int | 20 | 返回条数 |

```jsonc
// results[]（文档级，不含正文）
{ "doc_id":"aliyun_docs/zh/pai/billing-of-eas.md", "title":"EAS 计费说明",
  "product":"人工智能平台 PAI", "section":"计费",
  "source_url":"https://help.aliyun.com/zh/pai/billing-of-eas", "score":0.91 }
```

**模糊匹配实现**:数千文档用 `rapidfuzz`(`token_set_ratio`,O(N) 毫秒级);上万+用 trigram 倒排(`pg_trgm`/SQLite FTS5 trigram);中文按字符 bigram/trigram。字段加权 `title > path > summary`。

### 5.4 `fetch` — 按引用取全文 / 上下文

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `ref` | object | — | `{ doc_id }` 或 `{ chunk_id }` |
| `mode` | enum | `full_doc` | `full_doc` / `chunk_neighbors` / `section` |
| `window` | int | 1 | `chunk_neighbors` 时取前后各 N 块 |

| mode | 行为 | ref |
|------|------|-----|
| `full_doc` | 整篇 markdown | `doc_id`/`chunk_id` |
| `chunk_neighbors` | 目标切片 + 前后 `window` 块 | `chunk_id` |
| `section` | 切片所在 `heading_path` 那一节 | `chunk_id` |

```jsonc
{ "doc_id":"easyrec_docs/en/latest/quick_start.md", "title":"快速开始",
  "content":"# 快速开始\n\n...", "mode":"full_doc",
  "source_url":"https://easyrec.readthedocs.io/en/latest/quick_start.html",
  "metadata": { /* 完整 document metadata */ } }
```

---

## 6. 典型 Agent 工作流

```
模糊问题      → search(query, filters)            → 切片(带 metadata)
确切标识符    → keyword(pattern, scope)
"有没有X文档" → catalog(query="X 计费")
读全文/核对   → fetch(ref={chunk_id}, mode=section|full_doc)
```

迭代:`search` → 切片不够 → `fetch` 取上下文 → 不满意 → `keyword` 精确定位 → 再 `fetch`。

---

## 7. 待定 / 可选

- **分页**:大结果集是否需 `cursor`(当前 `limit` + `total`)。
- **多源联邦打分**:多 datasource 的 `search` 结果如何归一(统一 rerank 是自然解)。
- **权限维度**:不同 datasource 有访问控制时,`filters` 叠加权限。
- **版本**:同一文档多版本(如 readthedocs `latest`/`v0.5`)是否并存及如何选。
