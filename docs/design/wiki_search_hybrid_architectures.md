# Wiki + Search 混合架构方案对比

> **目标**: 对比主流 Wiki+Search 混合架构方案, 为 PAI-RAG 新一代知识系统设计提供决策依据  
> **日期**: May 2026  
> **背景**: Karpathy LLM Wiki (Apr 2026) 引爆了 "编译知识 vs 实时检索" 的讨论, 但纯 Wiki 有规模限制 (≤500页), 纯 RAG 有综合能力短板. 混合架构是 2026 工业共识.

---

## 一、核心命题: 推理在何时发生?

所有方案的根本区别在于 **重推理 (heavy reasoning) 放在哪个阶段**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         知识处理时间轴                               │
│                                                                     │
│  INGEST TIME          ─────────────────────>         QUERY TIME     │
│                                                                     │
│  ◀──── 编译型 (Wiki/RAPTOR/GraphRAG) ────▶                         │
│        重推理在摄入时, 查询轻量                                       │
│                                                                     │
│                        ◀──── 检索型 (RAG) ────▶                     │
│                        重推理在查询时, 摄入轻量                       │
│                                                                     │
│  ◀────────────── 混合型 (本文讨论) ──────────────▶                  │
│  摄入时编译核心知识, 查询时检索补充细节                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、六种主流方案深度对比

### 方案 A: Karpathy LLM Wiki (纯编译 + Index导航)

**架构:**
```
Raw Sources → LLM Compiler → Structured Wiki (Markdown)
                                    │
Query → Read index.md → Pick pages → Read pages → Synthesize Answer
```

**搜索机制:** Agent 读取 `index.md` (全页面目录), 通过 LLM 推理选择相关页面, 直接读取  
**关键特征:**
- 零基础设施 (无向量库, 无 embedding)
- 确定性检索 (文件级, 非概率性)
- Agent 可读可写 (自进化)
- Git 版本化

**实测数据 (arxiv:2605.18490):**
- 跨文档综合能力: **+6.63/10** (vs RAG)
- 引用精度: **40.2%** vs RAG 18.9%
- 查询 token 消耗: **21× 高于 RAG** (agent 多轮浏览)
- 延迟: 22 min vs RAG 3.3 min

**适用:** ≤100 文档, 稳定知识, 需要人类可审阅

---

### 方案 B: Cache-Augmented Generation (CAG) + KV Cache

**架构:**
```
Raw Sources → LLM Compile → Wiki Pages → Load into Context → Cache KV States
                                                                    │
Query → Append to cached context → Generate (sub-second TTFT)
```

**搜索机制:** 无搜索! 全部知识预加载到 context window, 通过 attention 机制隐式 "搜索"  
**关键特征:**
- 消除 retrieval failure (所有知识始终可见)
- Sub-second 延迟 (KV cache 命中后)
- 无 chunking、无 embedding drift

**实测数据 (WiCER paper, May 2026):**
- 编辑精选内容 (67K tokens, 30篇): Full-context **4.38/5** vs RAG **4.08/5**, TTFT 快 **7.3×**
- 原始文档 (80篇, 55-95K tokens): Full-context **3.47** vs RAG **3.64** (attention dilution!)
- 盲编译后: 灾难性降级到 **2.14-2.32** (LLM 过度压缩丢失关键事实)
- WiCER 迭代修复: 恢复 **80%** 质量差距

**规模上限:** ~96K-200K tokens (当前主流模型 context window)  
**核心风险:** Attention dilution — 文档过多时注意力分散, 质量反而低于 RAG  
**适用:** ≤30 篇精选文档, 高频查询 (amortize cache cost), 低延迟要求

---

### 方案 C: RAPTOR (递归抽象处理树)

**架构:**
```
Chunks → Embed → Cluster (k-means) → LLM Summarize → Embed summaries
         → Cluster again → Summarize → ... (recursive tree)

         Level 0: Raw chunks (leaf nodes)
         Level 1: Cluster summaries  
         Level 2: Higher-level summaries
         Level N: Root summary (entire corpus)

Query → Traverse tree (top-down) OR Search collapsed tree (all levels)
```

**搜索机制:** 两种策略:
1. **Tree Traversal**: 从根节点向下, 每层选最相关 top-k 子节点
2. **Collapsed Tree**: 把所有层级节点 flatten 到一个向量索引, 直接 top-k search

**关键特征:**
- 多粒度检索: 可以检索到 "单个段落" 或 "整个主题的综合摘要"
- 自然支持 global + local queries
- 兼容现有向量数据库 (所有节点都有 embedding)
- RAGFlow 已集成 RAPTOR 作为标准选项

**实测数据 (ICLR 2024):**
- 在 NarrativeQA 上提升 **20%** (vs flat chunking)
- 特别擅长 "thematic" 和 "跨文档" 类问题
- 构建成本: 每次 cluster+summarize 需要 N 次 LLM 调用

**适用:** 中等规模 (100-10K 文档), 需要多粒度回答, 兼顾 local/global query

---

### 方案 D: GraphRAG Community Summaries + Vector Hybrid

**架构:**
```
Documents → Entity Extraction → Knowledge Graph → Leiden Community Detection
                                                        │
                              ┌──────────────────────────┼──────────────────┐
                              │                          │                  │
                      Community Summaries         Entity Embeddings    Chunk Embeddings
                      (Global Search)             (Local Search)       (Local Search)
                              │                          │                  │
Query → Router → ┌─ Global: Search summaries → Map-Reduce → Answer
                 └─ Local:  Vector search entities/chunks → Graph traverse → Answer
```

**搜索机制:** 双模搜索:
1. **Local Search**: Vector search → entity/chunk → 图邻域扩展 (traverse related entities)
2. **Global Search**: 对 community summaries 做 map-reduce (预编译的主题综述)

**关键特征:**
- Community summaries **就是一种 Wiki 编译** (将实体集群编译为主题综述)
- Graph traversal 提供精确的关系推理
- 支持 "What are the main themes?" 类 high-level 问题
- Entity descriptions + community summaries 都有 vector index

**实测数据:**
- 在 synthesis 类问题上比 flat RAG 高 **~2×**
- 构建成本: ~$7 per moderate corpus (entity extraction 是大头)
- Global queries 秒级响应 (预计算的 community summaries)

**适用:** 关系密集型数据 (人物、组织、事件), 需要回答 global/thematic 问题

---

### 方案 E: Hybrid Wiki + RAG Router (Particula/业界推荐模式)

**架构:**
```
Sources ─┬─── Stable/Core → LLM Compile → Wiki Pages ──┐
         │                                              │
         └─── Dynamic/Large → Chunk + Embed → VectorDB ─┤
                                                        │
Query → Agent Router (LLM) ─┬─ "Refund policy?" → Wiki (deterministic, fast)
                            ├─ "Yesterday's ticket?" → RAG (fresh, dynamic)
                            └─ "Policy vs complaint?" → Wiki + RAG (both)
```

**搜索机制:** 路由 + 双通道:
1. Wiki 通道: Index navigation 或 BM25/vector over compiled pages
2. RAG 通道: Standard vector + rerank
3. Router: LLM-based 或 rule-based, 判断 query 适合哪个通道

**关键特征:**
- **编译层**: 核心/稳定知识 → Wiki (高质量综合, 可审阅)
- **检索层**: 动态/海量数据 → RAG (低延迟, 高规模)
- **路由层**: 按 query 特征选通道 (或两者都用)
- 实现相对简单 (两个独立系统 + 路由)

**实际案例:**
- Particula: 推荐为 "most teams should land on"
- Karpathy Gist 评论区: "semantic search index sits alongside the wiki"
- ex-brain (OceanBase): BM25+vector hybrid over compiled wiki pages

**适用:** 通用架构, 适合大多数企业场景

---

### 方案 F: Google NotebookLM / Code Wiki (Full-Context Grounding)

**架构:**
```
Sources → Upload → Gemini 1.5/2.0 ingests FULL sources into context window
                                    │
Query → Source Grounding (attention over full sources) → Cited Answer
         (no chunking, no embedding, no retrieval — direct attention)

Code Wiki variant:
Codebase → Gemini generates full wiki → Wiki always in context → Chat over it
           (regenerated per-commit)
```

**搜索机制:** 无显式搜索 — 依赖大 context window (2M tokens) + attention 机制  
**关键特征:**
- 零基础设施 (no vector DB, no chunking)
- 完美的跨文档推理 (全部在同一 context)
- Source Grounding: 每个回答锚定到具体源
- Code Wiki: 每次 commit 后重新生成整个文档

**规模上限:** ~2M tokens (~500页, Gemini 2.0)  
**核心风险:** "Lost in the Middle" — 中间部分信息可能被忽略  
**成本:** 每次查询需要处理全部 tokens (expensive at scale)

**适用:** 中等规模 (≤500页), Google 生态, 代码文档场景

---

## 三、全维度对比矩阵

| 维度 | A: Karpathy Wiki | B: CAG/KV Cache | C: RAPTOR | D: GraphRAG | E: Hybrid Router | F: NotebookLM |
|---|---|---|---|---|---|---|
| **搜索机制** | Index 导航 (LLM读目录选页) | 无 (attention 隐式) | Tree traverse / collapsed search | Vector + graph traverse | Router → Wiki or RAG | Attention over full context |
| **向量数据库** | ❌ 不需要 | ❌ 不需要 | ✅ 需要 (所有层级) | ✅ 需要 | ✅ RAG通道需要 | ❌ 不需要 |
| **摄入成本** | 高 ($0.15-0.30/doc) | 高 (compile + cache) | 高 (递归summarize) | 很高 (~$7/corpus) | 中 (部分compile) | 低-中 (upload) |
| **查询成本** | 高 (21× RAG) | 低 (cached) | 中 (vector search) | 中 | 中 | 高 (full context) |
| **查询延迟** | 4-8s | <1s (cached) | 200-500ms | 200-500ms | 1-5s avg | 2-8s |
| **规模上限** | ~100 docs (index瓶颈) | ~30 docs (context window) | 10K+ docs | 10K+ docs | 100K+ docs | ~500 pages (2M tokens) |
| **综合质量** | 极高 (+6.63/10) | 高 (curated时) | 高 (多粒度) | 高 (关系型) | 高 (核心wiki部分) | 高 |
| **引用精度** | 极高 (40.2%) | 高 | 中 | 中 | 混合 | 高 (source grounding) |
| **错误传播** | ⚠️ 编译错误永久化 | ⚠️ 同上 | ⚠️ 摘要可能丢信息 | ⚠️ 实体提取错误 | 部分隔离 | 低 (直接引用源) |
| **人类可读** | ✅ Markdown wiki | ❌ KV cache不可读 | ❌ 树结构不直观 | 部分 (community summaries) | ✅ Wiki部分 | ✅ 生成的wiki |
| **自进化** | ✅ Agent写回 | ❌ 需重建cache | ❌ 需重建树 | ❌ 需重建图 | 部分 | ✅ 每commit重生成 |
| **多模态** | ❌ (纯文本) | ❌ | ❌ | ❌ | ✅ RAG通道可支持 | ✅ Gemini原生 |
| **开源实现** | CacheZero, llmwiki, ex-brain | WiCER (研究) | RAGFlow内置 | Microsoft GraphRAG | 自建 | Google 专有 |
| **MCP暴露** | ✅ (llmwiki via MCP) | ❌ | ❌ (可自建) | ❌ (可自建) | ✅ (自然) | ❌ |

---

## 四、混合架构设计空间: 五个关键决策点

### Decision 1: Wiki 编译粒度

| 粒度 | 描述 | 代表 | 适用 |
|---|---|---|---|
| **Topic Page** | 每个主题一个综合页 (pricing-model.md) | Karpathy Wiki | 稳定领域知识 |
| **Community Summary** | 每个实体集群一个综述 | GraphRAG | 关系密集型 |
| **Hierarchical** | 多层摘要树 (chunk → cluster → topic → global) | RAPTOR | 多粒度需求 |
| **Full Compilation** | 整个语料编译为结构化wiki | WiCER | 高频查询, 小语料 |
| **Module-level** | 每个代码模块一张知识卡片 | Qoder RepoWiki, Google Code Wiki | 代码库 |

### Decision 2: Wiki 上的搜索层

| 策略 | 延迟 | 精度 | 规模 | 复杂度 |
|---|---|---|---|---|
| **Pure Index (LLM选页)** | 4-8s | 高 (LLM推理) | ≤100页 | 极低 |
| **BM25 over wiki pages** | <100ms | 中 | 1000+ | 低 |
| **Vector search over wiki pages** | <200ms | 高 | 10K+ | 中 |
| **Hybrid BM25+Vector over wiki** | <200ms | 很高 | 10K+ | 中 |
| **Full context (all pages in window)** | 2-8s | 最高 | ≤500页 | 极低 |
| **Tree traversal (RAPTOR)** | <300ms | 高 | 10K+ | 高 |

### Decision 3: RAG 兜底策略

```
Query → Wiki Search → Found? ─── Yes → Generate from wiki pages
                         │
                         No (或 confidence 低) → Fallback to RAG → Generate from raw chunks
```

**关键问题:** 何时 fallback?
- **Coverage-based**: wiki 没有相关页面时
- **Recency-based**: query 涉及近期数据 (wiki 可能未更新)
- **Detail-based**: query 需要原始细节 (wiki 摘要可能丢失)
- **Always-both**: 同时查 wiki + RAG, merge 结果 (最可靠但最贵)

### Decision 4: Wiki 更新/再生策略

| 策略 | 触发 | 延迟 | 一致性 | 代表 |
|---|---|---|---|---|
| **Manual** | 人工触发 | 按需 | 低 | Karpathy 原始 |
| **Per-document** | 新文档摄入时 | 分钟级 | 中 | Particula |
| **Per-commit** | 代码提交时 | 分钟级 | 高 | Google Code Wiki |
| **Scheduled** | 定时 (每天/每周) | 小时-天级 | 中 | 企业常见 |
| **Query-driven (WiCER)** | 高频问题暴露 wiki 缺陷时 | 按需 | 高 | WiCER paper |
| **Agent-driven** | Agent 发现矛盾/缺失时写回 | 实时 | 最高 | Letta, Karpathy 进阶 |

### Decision 5: 编译质量保证

| 机制 | 描述 | 代表 |
|---|---|---|
| **Lint pass** | 定期扫描矛盾、孤立页、陈旧信息 | Karpathy Wiki |
| **Diagnostic probes** | 用测试问题验证 wiki, 发现丢失事实 | WiCER |
| **Human review** | Wiki 页面由人工审核后发布 | 合规场景 |
| **Citation chain** | 每个 wiki 声明追溯到原始源, 可验证 | 所有方案应该都做 |
| **Conflict detection** | 新源与已有 wiki 页矛盾时告警 | Karpathy Wiki, WiCER |
| **Coverage metrics** | 追踪哪些源尚未被 wiki 覆盖 | 企业需求 |

---

## 五、我们的设计空间 (Brainstorming Seeds)

### 思路 1: "Knowledge Compiler as a Service" (编译器即服务)

```
┌─────────────────────────────────────────────────────────────┐
│  Knowledge Compiler                                          │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐ │
│  │ Topic Wiki   │  │ Entity Graph│  │ RAPTOR Tree        │ │
│  │ (Markdown)   │  │ (Summaries) │  │ (Multi-level)      │ │
│  └──────┬───────┘  └──────┬──────┘  └────────┬───────────┘ │
│         │                  │                   │             │
│         └──────────────────┼───────────────────┘             │
│                            │                                 │
│                   Unified Search Index                       │
│            (Vector + BM25 over ALL compiled artifacts)       │
└────────────────────────────┬────────────────────────────────┘
                             │
                     MCP Server Interface
                             │
                 ┌───────────┼───────────┐
                 │           │           │
           read_wiki()  search()   get_summary()
```

**核心想法**: Wiki pages、GraphRAG community summaries、RAPTOR 树节点 — 都是 "编译产物". 统一放到一个搜索索引里, 查询时不区分来源.

---

### 思路 2: "Tiered Compilation" (分层编译)

```
Tier 0: Core Knowledge (always compiled, always in context)
         → ≤10 pages, 最重要的领域知识
         → 类似 Letta Core Memory

Tier 1: Topic Wiki (compiled, searchable on demand)
         → 50-200 topic pages
         → Hybrid BM25+Vector search
         → Agent 可读可写

Tier 2: Raw Chunks (not compiled, standard RAG)
         → 10K+ chunks from original documents
         → Standard vector search + rerank
         → 用于 detail lookup / 最新数据

Tier 3: External Sources (live retrieval)
         → Web search, API calls
         → 用于 wiki/RAG 都没有的情况
```

Query → Complexity Router:
- Simple known topic → Tier 0 (zero retrieval)
- Standard Q&A → Tier 1 (wiki search)
- Detail/quote needed → Tier 2 (RAG)
- Unknown/live data → Tier 3 (external)
- Complex research → All tiers + multi-hop

---

### 思路 3: "Living Wiki with RAG Grounding" (活 Wiki + RAG 事实验证)

```
Write path:
  New Doc → RAG Ingest (chunk+embed) → also trigger Wiki Update (LLM reads new doc + existing wiki → updates relevant pages)

Read path:
  Query → Search Wiki → Draft Answer (from compiled knowledge)
        → Verify: RAG search for supporting raw chunks
        → If contradiction found: flag & show both versions
        → If no support found: fall back to RAG-only answer
```

**核心想法**: Wiki 提供综合 (synthesis), RAG 提供验证 (grounding). 
- Wiki answers = "这是我们理解的..."
- RAG verification = "以下原文支持/反对这个理解..."
- 这解决了 Wiki 的 "error amplification" 问题

---

### 思路 4: "Decomp-RAG as the Middle Ground" (分解式 RAG 作为折中)

来自 arxiv:2605.18490 的发现: Decomp-RAG 能恢复 Wiki 88% 的综合优势, 只需 3.4× 成本 (vs Wiki 的 21×).

```
Query → Decompose into sub-questions (LLM)
      → Per sub-question: retrieve + validate (existing RAG)
      → Merge: deduplicate chunks across sub-questions
      → Synthesize: structured answer from merged context
```

**无需 Wiki 编译层**, 只需在查询时更聪明. 但:
- 不产生可审阅的知识库
- 不能自进化
- 每次查询重新推导

可作为 "wiki 尚未覆盖的主题" 的 fallback 策略.

---

### 思路 5: "Agent-Maintained Wiki + Search" (Agent自维护Wiki)

```
Background Agent (sleep-time compute):
  - Periodically scans RAG index for high-frequency topics
  - Compiles wiki pages for popular topics (demand-driven)
  - Detects stale wiki pages (source updated but wiki hasn't)
  - Runs lint: contradiction check, coverage check
  - Merges user feedback into wiki

Query Agent:
  - Reads wiki index → if topic exists → wiki answer
  - If topic missing → RAG answer + suggest "compile this topic?"
  - If answer uncertain → RAG verify wiki claims
```

**核心想法**: Wiki 不是一次性编译, 而是 Agent 持续维护的产物. 类似 Letta 的 "sleep-time compute" — 在无查询时做知识整理.

---

## 六、我的推荐: 对 PAI-RAG 新系统的方案建议

基于以上分析, 推荐采用 **思路 2 (Tiered Compilation) + 思路 3 (RAG Grounding) + 思路 5 (Agent Maintenance)** 的组合:

```
┌────────────────────────────────────────────────────────────────────┐
│                    PAI-RAG Context Engine v2                        │
│                                                                    │
│  Query → Complexity Router → ┌─ Tier 0: Core (injected)           │
│                              ├─ Tier 1: Wiki (compiled, searchable)│
│                              ├─ Tier 2: RAG (chunks, standard)    │
│                              └─ Tier 3: Research (multi-hop)      │
│                                                                    │
│  Wiki Layer:                                                       │
│    - Markdown pages (human-readable, git-versioned)                │
│    - Hybrid BM25+Vector search over wiki pages                     │
│    - Agent-maintained (background compilation + lint)              │
│    - RAG grounding (verify wiki claims against raw sources)        │
│    - MCP exposure: read_wiki(), search_wiki(), list_topics()       │
│                                                                    │
│  RAG Layer:                                                        │
│    - Standard chunk+embed+vector search                            │
│    - Serves as: (a) detail lookup (b) wiki verification            │
│    - Also feeds wiki compilation (source material)                 │
│                                                                    │
│  Update Strategy:                                                  │
│    - New document → RAG ingest (immediate)                         │
│    - New document → Wiki update (async, agent-driven)              │
│    - Query feedback → Wiki refinement (WiCER-inspired)             │
│    - Scheduled lint → Contradiction/staleness detection            │
└────────────────────────────────────────────────────────────────────┘
```

**Why this combination:**
1. **Wiki 提供综合能力** (Karpathy 的核心洞察: 编译一次, 复用多次)
2. **RAG 提供规模和验证** (Wiki 的 500页上限由 RAG 兜底)
3. **Agent 维护保鲜** (不依赖人工, 自动检测陈旧/矛盾)
4. **Hybrid search 解决 Wiki 导航瓶颈** (不再 21× token, 而是 <200ms vector search)
5. **MCP 暴露** 让任何 agent 框架都能用

---

## 七、Open Questions for Brainstorming

1. **编译粒度**: 一个 topic = 一个 wiki page? 还是按 entity/concept/FAQ 分更细的 page?

2. **Wiki 与 GraphRAG 的关系**: Community summaries 本质就是 wiki pages. 是统一为一种 (markdown pages with entity tags), 还是分开两套系统?

3. **触发编译的条件**: 
   - 新文档进来就编译相关 wiki? (及时但贵)
   - 等查询失败时才编译? (省钱但体验差)
   - 后台 agent 周期性扫描 "热度" 来决定? (平衡方案)

4. **Wiki 的 "真相源" 问题**: Wiki page 说 A, 原始文档说 B — 谁赢? 
   - 选项1: 原文永远赢 (wiki 只是辅助)
   - 选项2: Wiki 赢 (除非 lint 发现矛盾) 
   - 选项3: 展示两者 + conflict flag

5. **用户可编辑 Wiki?** 如果让人类编辑 wiki pages, 如何与 LLM 自动编译共存? (Conflict resolution)

6. **多语言**: Wiki pages 的语言? 跟源文档一致? 还是统一编译为一种语言?

7. **成本模型**: 编译一个 100 文档的知识库 → wiki 需要多少 LLM 调用? 持续维护的月成本?
