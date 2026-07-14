# 知识库在线搜索

本文说明知识库在线搜索的生产默认值、候选数量、rerank 顺序、降级行为和 Trace 结构。离线导入与索引流程不在本文范围内。

## 默认行为

Web 控制台固定使用 `hybrid` 搜索，不向普通用户展示 `hybrid`、`vector`、`keyword` 模式选择。后端 API 和搜索引擎接口仍保留三种模式，用于兼容、测试和诊断；未指定时默认为 `hybrid`。

Agent 的 `knowledge_search` 默认 `top_k=10`。Agent 可以关联多个知识库，服务会分别召回候选，合并后统一 rerank，而不是逐个知识库独立 rerank。

召回测试页走同一检索服务，但不读取 Agent 的 rerank 配置，因此展示的是搜索引擎产生的最终检索分数。页面只展示 `score`，不拆分 BM25、向量或混合分数。

## 参数关系

### `top_k`

`top_k` 是分页后最终返回的切片数量：

- 默认值：10
- 后端有效范围：1～50
- 分页窗口：`top_k + offset`

### 每知识库召回窗口

每个允许访问的知识库独立召回：

```text
per_kb_fetch_limit = max(20, top_k + offset)
```

即使某个知识库很大，也不能在检索请求阶段挤掉其他知识库的全部候选。多个知识库的 `total` 会相加，因此日志中的 `total` 可能明显大于实际进入 rerank 的候选数量；它表示各知识库匹配总数之和，不是候选池大小。

### `candidate_pool_size`

`candidate_pool_size` 是 Agent rerank 配置，默认 50，配置范围为 1～200。它只在 Agent 启用 rerank 且 rerank provider 可用时生效。

```text
effective_candidate_pool_size = max(candidate_pool_size, top_k + offset)
```

使用分页窗口作为下限，避免较小配置导致当前页没有足够候选。候选不足时使用全部候选，不填充、不重复。

## 执行顺序

在线搜索按以下顺序执行：

1. 根据当前用户权限过滤知识库。
2. 按 embedding provider、model 和 dimension 对知识库分组。
3. `vector` 或 `hybrid` 模式下，每个 embedding 配置组生成一次 query embedding。
4. 每个知识库独立召回 `per_kb_fetch_limit` 个候选。
5. 合并全部知识库候选，按搜索引擎 `score` 降序排列。
6. 启用 Agent rerank 时，同一 `(kb_id, document_id)` 最多保留 3 个切片；缺少 `document_id` 时使用 `chunk_id`。
7. 截取前 `effective_candidate_pool_size` 个候选。
8. 将全部知识库候选统一送入一次 reranker。输入包含知识库名称、文档标题、heading 和切片正文。
9. 对统一排序结果应用 `offset:offset + top_k`。

rerank 成功时，结果 `score` 被替换为 rerank relevance。rerank 关闭时，第 6～8 步全部跳过，结果直接按搜索引擎得分分页。

## Elasticsearch hybrid

Elasticsearch 的 `hybrid` 是一次 `_search` 请求，同时包含：

- `multi_match` BM25：搜索 `text`、`title^2` 和 `heading^1.5`。
- `dense_vector` kNN：使用当前 embedding 配置生成的 query vector。

请求窗口为：

```text
size = limit
k = offset + limit
num_candidates = max(50, (offset + limit) * 4)
```

`KnowledgeService` 对每个知识库调用时使用 `offset=0`、`limit=per_kb_fetch_limit`。例如每知识库召回 20 条时，Elasticsearch 收到 `size=20`、`k=20`、`num_candidates=80`。

BM25 和 kNN 由 Elasticsearch 在同一请求中合并 score。应用层不拆成两次查询，不执行额外 RRF 或其他融合。

## Rerank 与候选多样化

候选多样化和 rerank 不能互相替代：

- 每文档最多 3 个切片用于控制候选覆盖面，防止一篇长文档占满 rerank 窗口。
- rerank 判断已经进入窗口的切片与 query 的相关性。

如果其他文档在 rerank 前已经被挤出窗口，reranker 无法恢复它们。因此处理顺序必须是“按检索得分合并 → 每文档限制 → candidate pool → rerank”。

## 降级与错误

- 空 query 或没有可访问知识库：返回空结果。
- query embedding 失败：记录错误类型并继续现有搜索引擎路径。
- 主搜索引擎失败：启用本地降级时改用 `LocalSearchEngine`；禁用时向上抛错。
- rerank 模型缺失：保留截断后的检索顺序。
- reranker 调用失败或返回空结果：保留截断后的检索顺序。
- Trace 扩展缺失、未初始化或关闭：所有 Trace helper 退化为 no-op，不改变搜索结果。

日志和 Trace 不记录原始 query、切片正文、标题或异常字符串。失败仅记录受控状态和异常类型，例如 `RuntimeError`。

## Trace

Agent 调用时，搜索 span 位于已有的 `tool knowledge_search` span 下；通过 HTTP 直接测试时，它位于当前请求 span 下。

```text
tool knowledge_search
└── knowledge.search
    ├── knowledge.query_embedding
    ├── knowledge.retrieve.hybrid
    │   ├── event: bm25
    │   └── event: vector_knn
    ├── knowledge.candidate_diversify
    └── knowledge.rerank
```

每个 embedding 配置组产生一个 `knowledge.query_embedding`。每个知识库产生一个检索 span：

- `hybrid`：`knowledge.retrieve.hybrid`
- `keyword`：`knowledge.retrieve.bm25`
- `vector`：`knowledge.retrieve.vector`

hybrid 中 BM25 和 kNN 是同一个 Elasticsearch 请求，无法获得可信的独立耗时。因此它们表现为 `knowledge.retrieve.hybrid` 下的 `bm25` 和 `vector_knn` 事件，而不是两个伪造的子 span。

关键属性包括：

- 根 span：mode、`top_k`、offset、知识库数、rerank 状态、configured/effective candidate pool、候选数、结果数和匹配总数。
- embedding：provider、model、dimension 和共享该向量的知识库数。
- retrieval：知识库 ID、执行引擎、召回 limit、结果数、匹配总数、是否降级，以及 kNN 的 `k`、`num_candidates`。
- candidate diversify：输入、输出、丢弃数和每文档最大切片数。
- rerank：model、输入数、`top_n`、输出数和执行状态。

## 调优示例

假设 Agent 关联两个知识库，配置如下：

```text
top_k = 10
offset = 0
candidate_pool_size = 50
max_chunks_per_document = 3
```

执行数量为：

1. 每个知识库分别请求 20 个候选，最多合并 40 个实际候选。
2. 按检索分数合并，再将每篇文档限制为最多 3 个切片。
3. effective candidate pool 为 `max(50, 10)=50`；如果多样化后只有 32 个候选，则 32 个全部进入一次 rerank。
4. rerank 后返回前 10 个切片。

如果关联 5 个知识库，每个知识库都返回 20 个候选，则合并前最多 100 个；多样化后只取前 50 个进入 rerank。增大 `candidate_pool_size` 可以提高覆盖面，但会增加 rerank 延迟和调用成本。
