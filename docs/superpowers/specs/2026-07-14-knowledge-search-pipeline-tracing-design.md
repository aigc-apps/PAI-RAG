# 知识库搜索候选池与链路追踪设计

日期：2026-07-14

## 背景与目标

当前知识库搜索已经按知识库分别召回候选，并支持在 Agent 层配置统一 rerank，但仍有三个问题：

1. Agent 的 `candidate_pool_size` 已进入配置和调用链，却没有真正限制 rerank 输入数量。
2. 前端仍暴露 `hybrid`、`vector`、`keyword` 模式，增加了普通用户不必要的选择；生产默认应直接使用 Elasticsearch hybrid 检索。
3. 日志只能看到聚合后的候选数量，无法在 Langfuse/OpenTelemetry 中定位向量生成、检索、候选去重和 rerank 的耗时与降级行为。

本次改动使生产搜索路径固定、可解释、可观测，同时保留后端模式参数用于 API 兼容、测试和诊断。

## 范围

本次包含：

- 前端移除搜索模式选择和模式展示，前端请求固定使用 `hybrid`。
- 后端继续接受 `hybrid`、`vector`、`keyword`，默认值仍为 `hybrid`。
- 修复 Agent rerank 的 `candidate_pool_size` 行为。
- 保留 rerank 前的单文档候选上限，并明确它与 rerank 的职责边界。
- 为在线搜索的关键阶段增加 OpenTelemetry span 和属性。
- 增加面向维护者的知识库搜索配置与执行流程文档。

本次不包含：

- 将 Elasticsearch hybrid 拆成两次独立的 BM25 和 kNN 请求。
- 引入 RRF 或改变 Elasticsearch 当前的原生混合打分方式。
- 将搜索模式从后端 API、工具协议或搜索引擎接口中删除。
- 新增可配置的单文档切片上限；本次继续固定为每文档最多 3 个候选切片。

## 搜索数据流

### 1. 参数归一化

- `top_k` 是最终返回数量，默认 10，后端限制为 1～50。
- `offset` 用于全局分页，最小为 0。
- 每个知识库独立召回 `max(20, top_k + offset)` 个候选，使不同规模的知识库都有进入统一排序的机会。
- 当 Agent rerank 未启用或 rerank 运行条件不满足时，`candidate_pool_size` 不参与搜索。

### 2. 按 embedding 配置生成查询向量

允许搜索的知识库按 embedding provider、model 和 dimension 分组。`vector` 或 `hybrid` 模式下，每组只生成一次 query embedding，并复用于组内各知识库。

embedding 失败时保留现有 best-effort 行为：记录错误类型并继续调用搜索引擎。搜索引擎可以按现有实现降级或返回可用的关键词结果，不把原始 query 写入 span 属性。

### 3. 每个知识库独立召回

每个知识库分别调用当前主搜索引擎，召回上一步计算的数量。Elasticsearch 的 `hybrid` 仍由一次 `_search` 请求同时携带 BM25 `query` 和 kNN 子句：

- BM25 和 kNN 使用同一个 Elasticsearch 请求窗口。
- kNN 的 `k` 为 `offset + limit`；本流程对单知识库调用时 offset 为 0，因此等于该知识库的召回数量。
- `num_candidates` 为 `max(50, k * 4)`。
- Elasticsearch 按现有原生 score 合并两个子句，不在应用层再次融合。

主搜索引擎失败且允许本地降级时，继续使用 `LocalSearchEngine`，并在 trace 中记录降级引擎和错误类型。禁用降级时维持抛错行为。

### 4. 合并与候选多样化

所有知识库的结果按原始检索 `score` 降序合并。启用 rerank 时，先执行候选多样化：同一 `(kb_id, document_id)` 最多保留 3 个切片；没有 `document_id` 时使用 `chunk_id` 作为兜底键。

该步骤不能由 rerank 替代。rerank 对收到的切片逐条相关性排序，但无法恢复在候选窗口中已经被某篇长文档挤掉的其他文档。候选多样化负责保证文档覆盖面，rerank 负责判断保留下来的切片相关性，两者解决不同问题。

### 5. 候选池截断与统一 rerank

候选多样化后，按以下规则计算 rerank 输入上限：

```text
effective_candidate_pool_size = max(configured_candidate_pool_size, top_k + offset)
```

`configured_candidate_pool_size` 来自 Agent 配置，默认 50，模型校验范围为 1～200。使用 `top_k + offset` 作为下限，避免配置较小时破坏分页或导致无法返回请求数量。随后按当前检索分数取前 `effective_candidate_pool_size` 个候选，并对来自全部知识库的候选只调用一次 reranker。

rerank 输入文本继续包含知识库名称、文档标题、heading 和正文。rerank 成功后用 rerank relevance 覆盖结果的最终 `score`；失败、模型缺失或 provider 不可用时，保留截断后的原始检索顺序。最后统一应用 `offset:offset + top_k`。

未启用 rerank 时，不执行文档候选上限和候选池截断，直接按合并后的检索分数分页。

## 前端行为

知识库设置页和召回测试页不再展示 `hybrid`、`vector`、`keyword` 选择器，也不在结果数量或空状态文案中展示、建议切换检索模式。

前端发起召回测试或保存默认检索配置时固定使用 `hybrid`，避免旧配置中的其他 mode 继续影响普通 UI。后端字段、类型和校验继续保留，便于内部 API 调用、测试与故障诊断。

召回测试继续只展示每条结果的最终 `score`。该页面不应用 Agent rerank 配置，因此展示搜索引擎产生的相似度/混合分数，不拆分关键词、向量或融合分数。

## Trace 设计

### Span 层级

```text
tool knowledge_search                         # Agent 工具层已存在
└── knowledge.search                          # KnowledgeService 搜索全流程
    ├── knowledge.query_embedding             # 每个 embedding 配置组一个
    ├── knowledge.retrieve.hybrid             # 每个 KB 一个；单次 ES 请求
    │   ├── event: bm25
    │   └── event: vector_knn
    ├── knowledge.candidate_diversify          # 启用 rerank 时
    └── knowledge.rerank                       # 启用 rerank 且存在候选时
```

后端显式使用纯模式时，检索 span 分别命名为 `knowledge.retrieve.bm25` 和 `knowledge.retrieve.vector`。对于 hybrid，BM25 与 kNN 在同一次 Elasticsearch 请求中执行，无法得到可信的独立耗时，因此只创建一个 `knowledge.retrieve.hybrid` span，并用两个事件说明请求包含的检索组件，不伪造独立子 span。

本地搜索和 Elasticsearch 降级路径沿用相同的逻辑 span 名称，通过 `knowledge.engine` 和 `knowledge.fallback` 属性区分执行引擎。

### 主要属性

`knowledge.search`：

- `knowledge.mode`
- `knowledge.top_k`
- `knowledge.offset`
- `knowledge.kb_count`
- `knowledge.rerank.enabled`
- `knowledge.rerank.model`（配置存在时）
- `knowledge.candidate_pool.configured`
- `knowledge.candidate_pool.effective`
- `knowledge.candidates.retrieved`
- `knowledge.results.count`
- `knowledge.total_hits`

`knowledge.query_embedding`：

- provider、model、dimension
- 共享该向量的知识库数量
- 成功、失败以及失败错误类型

`knowledge.retrieve.*`：

- `knowledge.kb_id`
- `knowledge.engine`
- `knowledge.retrieval.limit`
- `knowledge.results.count`
- `knowledge.total_hits`
- `knowledge.fallback`
- Elasticsearch hybrid 下的 `knowledge.knn.k` 和 `knowledge.knn.num_candidates`

`knowledge.candidate_diversify`：

- `knowledge.candidates.input`
- `knowledge.candidates.output`
- `knowledge.candidates.dropped`
- `knowledge.max_chunks_per_document=3`

`knowledge.rerank`：

- rerank model
- 输入候选数、请求 top_n 和输出候选数
- 成功、跳过或失败回退状态
- 失败错误类型

属性不得包含原始 query、切片正文、标题或其他可能泄露知识库内容的数据。异常记录只使用异常类型和受控状态，不把异常字符串作为默认属性。

### 可选依赖与状态

追踪必须保持可选。trace 扩展或 OpenTelemetry 不可用、未初始化或禁用时，搜索行为与性能路径仍可正常运行，不因观测代码导入失败。异常 span 标记 error 后继续遵循原有业务回退或抛错规则；观测代码本身不得改变搜索结果。

## 错误与边界情况

- 空 query 或无可访问知识库：返回空结果；`knowledge.search` 仍可记录零结果，不创建无意义的子阶段 span。
- `top_k`、`offset` 非法：继续使用现有归一化与 API 校验行为。
- `candidate_pool_size < top_k + offset`：effective pool 自动提升到分页所需大小。
- 候选少于 effective pool：全部送入 rerank，不填充也不重复。
- 单文档没有 `document_id`：以 `chunk_id` 区分，避免误删不同未知文档的候选。
- query embedding 失败：记录失败状态，继续现有搜索引擎路径。
- Elasticsearch 失败：配置允许时回退本地搜索；否则向上抛错。
- rerank 模型缺失或 reranker 调用失败：保留已多样化并截断的检索顺序，最终 score 保持检索得分。
- reranker 返回无效或部分索引：沿用 rerank 适配层的有效结果处理，并确保不会越界；未被 reranker 返回的候选不凭空补入最终高位。
- rerank 关闭：不应用 `candidate_pool_size` 或每文档 3 切片限制，保证原有纯检索语义。

## 测试策略

后端单元测试覆盖：

- 每个知识库至少召回 20 个候选。
- 多知识库结果在一次 rerank 前合并。
- 单文档最多 3 个切片后再应用 `candidate_pool_size`。
- effective pool 不小于 `top_k + offset`。
- rerank 关闭时不应用候选池和文档切片限制。
- rerank 失败保留截断后的检索排序。
- Elasticsearch hybrid 请求的 `size`、kNN `k` 和 `num_candidates`。
- trace 开启时 span 名称、层级和关键计数属性正确。
- trace 扩展不可用或禁用时搜索仍正常。
- hybrid 只产生一个 retrieval span，并包含 BM25/kNN 事件。

前端测试覆盖：

- 知识库设置和召回测试不再渲染模式选择器。
- 召回请求固定发送 `hybrid`。
- 结果和空状态文案不再显示检索模式。
- 召回结果只展示最终分数。

文档验证覆盖默认值、处理顺序、分页下限、rerank 开关语义、Elasticsearch kNN 候选计算方式以及 trace 字段说明。

## 验收标准

1. Agent 配置的 `candidate_pool_size` 实际限制统一 rerank 的输入数量，同时不小于 `top_k + offset`。
2. 每个知识库独立召回至少 20 个候选；统一 rerank 前每文档最多保留 3 个切片。
3. Elasticsearch 默认继续执行单请求 hybrid 检索，未引入两次查询或应用层融合。
4. 普通前端不再展示或切换检索模式，后端兼容能力保留。
5. Langfuse/OpenTelemetry 能看到 embedding、每 KB 检索、候选多样化和 rerank 阶段；hybrid 的 BM25/kNN 表达不产生误导性耗时。
6. 追踪关闭或依赖缺失不会影响搜索功能。
7. 用户文档能够明确解释 `top_k`、`candidate_pool_size`、每 KB 召回窗口和 rerank 的完整关系。
