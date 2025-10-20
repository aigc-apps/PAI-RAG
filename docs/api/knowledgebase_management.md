
# 📘 知识库管理 API 文档

> **基础路径**：`/v1/config/knowledgebases`  
> **说明**：用于创建、查询、更新和删除知识库（Knowledge Base），每个知识库可配置分块策略、检索参数和嵌入模型。

---

## 1. 创建知识库

**POST** `/v1/config/knowledgebases`

### 请求体（JSON）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | string | ✅ | - | 知识库名称（仅支持字母、数字、下划线和短横线，≤100字符） |
| `description` | string | ❌ |  | 知识库描述 |
| `embedding_model` | string | ✅ | - | 嵌入模型名称（必须为系统支持的模型） |
| `chunk_config` | object | ❌ | 见下表 | 分块配置 |
| `retrieval_config` | object | ❌ | 见下表 | 检索配置 |

#### `chunk_config`（可选）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | int | `1024` | 文本分块大小（字符数） |
| `chunk_overlap` | int | `50` | 相邻块重叠字符数 |
| `parser_type` | string | `"sentence"` | 解析器类型 |
| `separator` | string | `"\n\n"` | 分隔符（用于自定义分块） |

#### `retrieval_config`（可选）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retrieval_mode` | string | `"hybrid"` | 检索模式（`"vector"`, `"keyword"`, `"hybrid"`） |
| `top_k` | int | `5` | 返回最相似的 top-k 结果 |
| `similarity_threshold` | float | `0.5` | 相似度阈值（低于则过滤） |
| `vector_weight` | float | `0.5` | 向量检索权重（仅 hybrid 模式有效） |
| `enable_rerank` | bool | `false` | 是否启用重排序 |
| `rerank_model` | string | `""` | 重排序模型名称（若启用） |

### 响应

- **成功（200）**：
  ```json
  {
    "code": 200,
    "message": "知识库创建成功。",
    "data": {
      "id": "kb123...",
      "name": "my_kb",
      "description": "...",
      "embedding_model": "text-embedding-ada-002",
      "chunk_config": { ... },
      "retrieval_config": { ... },
      "created_at": "2025-10-17T10:00:00",
      "updated_at": "2025-10-17T10:00:00"
    }
  }
  ```

- **失败（400）**：
  - 名称已存在 → `"创建知识库失败: 知识库名称已存在。"`
  - 名称非法 → `"知识库名称不能为空。"` / `"只能包含字母、数字和下划线。"`
  - 缺少 embedding_model → `"需要提供Embedding模型才能创建知识库。"`

---

## 2. 获取知识库列表

**GET** `/v1/config/knowledgebases`

### 查询参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | `1` | 页码（≥1） |
| `size` | int | `10` | 每页数量（≤1000） |

> ⚠️ 系统保留知识库 `default_attachments` 不会出现在列表中。

### 响应（200）

```json
{
  "code": 200,
  "message": "获取知识库列表成功",
  "data": {
    "items": [ /* KbEntity 数组 */ ],
    "total": 15,
    "pages": 2,
    "page": 1,
    "size": 10
  }
}
```

---

## 3. 获取单个知识库详情

**GET** ``/v1/config/knowledgebases/{kb_id}`

### 路径参数

| 参数 | 说明 |
|------|------|
| `kb_id` | 知识库 ID（以 `kb` 开头的 UUID） |

### 响应

- **成功（200）**：返回完整的 `KbEntity`
- **失败（404）**：`"查询知识库失败: 知识库'xxx'不存在。"`

---

## 4. 更新知识库

**PUT** `/{kb_id}`

### 路径参数

| 参数 | 说明 |
|------|------|
| `kb_id` | 知识库 ID |

### 请求体（JSON）

与 **创建接口** 的请求体结构相同，**所有字段均为可选**。  
未提供的字段将保留原值。

> ✅ 支持部分更新（例如只改 `description`）。

### 响应

- **成功（200）**：返回更新后的 `KbEntity`
- **失败（404）**：知识库不存在
- **其他错误**：返回异常堆栈（开发调试用）

---

## 5. 删除知识库

**DELETE** ``/v1/config/knowledgebases/{kb_id}`

### 路径参数

| 参数 | 说明 |
|------|------|
| `kb_id` | 知识库 ID |

### 行为说明

删除操作不会删除背后的向量存储库，您可以手动在向量库管理界面删除。

### 响应

- **成功（200）**：
  ```json
  { "code": 200, "message": "知识库'kb123...'删除成功。" }
  ```
- **失败（404）**：知识库不存在

---

## 🔐 错误码汇总

| HTTP 状态码 | 错误码 | 说明 |
|------------|--------|------|
| 400 | - | 请求参数错误（如名称非法、缺少必填字段） |
| 404 | - | 知识库不存在 |
| 500 | - | 内部服务异常（如数据库错误） |

---
