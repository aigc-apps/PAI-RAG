

# 知识库文件管理 API 文档

> 本文档描述了知识库中文件的上传、查询、更新来源、删除及检索等操作接口。

---

## 认证方式

所有请求需在请求头中携带认证 Token：

```http
Authorization: EAS_TOKEN
```

> 🔐 请确保 `EAS_TOKEN` 安全存储，不可泄露。

---

## 1. 上传文件

将本地文件上传至指定知识库，支持保留目录结构。

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/config/knowledgebases/{knowledge_id}/files`
- **内容类型**：`multipart/form-data`

### 路径参数

| 参数名         | 类型   | 说明 |
|----------------|--------|------|
| `knowledge_id` | string | 知识库唯一 ID（如 `kb9acd1faa893d41b9b5487abe9105b7c7`） |

### 表单字段（form-data）

| 字段名  | 类型     | 必填 | 说明 |
|---------|----------|------|------|
| `files` | file     | 是   | 要上传的文件。可通过 `filename` 指定路径以保留目录结构（如 `test/pairag.md`） |

> ⚠️ **注意**：
> - 文件名在同一个知识库内必须唯一。
> - 若上传同名文件，则视为**更新该文件内容**（覆盖原文件并触发重新处理）。

### 示例请求

```bash
curl -X POST \
  "http://API_ENDPOINT/v1/config/knowledgebases/kb9acd1faa893d41b9b5487abe9105b7c7/files" \
  -H "Authorization: EAS_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F 'files=@/Users/feiyue/Documents/test_files/pairag.md;filename=test/pairag.md'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "文件上传成功",
  "data": [
    {
      "id": "b9b1147c58384e7f891752ea5adc4c35",
      "kb_id": "kb9acd1faa893d41b9b5487abe9105b7c7",
      "file_name": "test/pairag.md",
      "file_path": "kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md",
      "file_extension": ".md",
      "file_size": 555,
      "file_md5": "c2b99342e160c73f0cf2a2058d031870",
      "status": "pending",
      "failed_reason": null,
      "active": true,
      "created_at": "2025-08-14T03:23:52.569652",
      "updated_at": "2025-08-14T03:23:52.569667",
      "file_metadata": {
        "file_path": "kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md",
        "file_name": "test/pairag.md",
        "file_size": 555,
        "file_extension": ".md"
      }
    }
  ]
}
```

### 响应字段说明

| 字段名             | 类型     | 说明 |
|--------------------|----------|------|
| `id`               | string   | 文件唯一 ID，用于后续操作 |
| `status`           | string   | 处理状态（见下表） |
| `file_md5`         | string   | 文件内容 MD5 校验值 |
| `file_metadata`    | object   | 文件元信息，包含路径、大小等 |

#### 文件处理状态（`status`）

| 状态        | 说明 |
|-------------|------|
| `pending`     | 已上传，等待处理 |
| `parsing`     | 正在解析文件内容 |
| `persisting`  | 正在持久化（含向量化） |
| `succeeded`   | 处理成功，可检索 |
| `failed`      | 处理失败，`failed_reason` 中包含错误信息 |

---

## 2. 查询文件状态（按 ID）

获取单个文件的详细信息及其处理状态。

### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/knowledgebases/{knowledge_id}/files/{file_id}`

### 路径参数

| 参数名         | 类型   | 说明 |
|----------------|--------|------|
| `knowledge_id` | string | 知识库 ID |
| `file_id`      | string | 文件唯一 ID |

### 示例请求

```bash
curl -X GET \
  "http://API_ENDPOINT/v1/config/knowledgebases/kb9acd1faa893d41b9b5487abe9105b7c7/files/b9b1147c58384e7f891752ea5adc4c35" \
  -H "Authorization: EAS_TOKEN"
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "查询知识库文件成功。",
  "data": {
    "id": "b9b1147c58384e7f891752ea5adc4c35",
    "kb_id": "kb9acd1faa893d41b9b5487abe9105b7c7",
    "file_name": "test/pairag.md",
    "file_path": "kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md",
    "file_extension": ".md",
    "file_size": 555,
    "file_md5": "c2b99342e160c73f0cf2a2058d031870",
    "status": "succeeded",
    "failed_reason": null,
    "active": true,
    "created_at": "2025-08-14T03:23:52.569652",
    "updated_at": "2025-08-14T03:23:52.569667",
    "file_metadata": {
      "file_path": "kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md",
      "file_name": "test/pairag.md",
      "file_size": 555,
      "file_extension": ".md",
      "file_url": "localdata/knowledgebase/kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md"
    },
    "file_source": null
  }
}
```

---

## 3. 按文件名查询

通过文件名精确查找文件（适用于已知路径/名称的场景）。

### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/knowledgebases/{knowledge_id}/files`
- **查询参数**：`file_name=test/pairag.md`

### 示例请求

```bash
curl -X GET \
  "http://API_ENDPOINT/v1/config/knowledgebases/kb9acd1faa893d41b9b5487abe9105b7c7/files?file_name=test/pairag.md" \
  -H "Authorization: EAS_TOKEN"
```

> ✅ 响应格式与 [按 ID 查询](#成功响应-200-ok-1) 完全一致。

---

## 4. 删除文件

从知识库中删除指定文件。

### 请求信息

- **方法**：`DELETE`
- **路径**：`/v1/config/knowledgebases/{knowledge_id}/files/{file_id}`

### 示例请求

```bash
curl -X DELETE \
  "http://API_ENDPOINT/v1/config/knowledgebases/kb9acd1faa893d41b9b5487abe9105b7c7/files/b9b1147c58384e7f891752ea5adc4c35" \
  -H "Authorization: EAS_TOKEN"
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "删除知识库文件成功。",
  "data": [null]
}
```

> ⚠️ 删除后文件不可恢复，且将从向量库中移除。

---

## 5. 更新文件来源链接

为文件设置外部源链接（如语雀文档地址），便于跳转查看原文。

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/config/knowledgebases/{knowledge_id}/files/{file_id}/source`
- **内容类型**：`application/json`

### 请求体（Body）

```json
{
  "file_source": "https://aliyuque.antfin.com/pai/arch/ktarnwbn0iqpgy3b"
}
```

| 字段名        | 类型   | 必填 | 说明 |
|---------------|--------|------|------|
| `file_source` | string | 是   | 外部文档链接（URL） |

### 示例请求

```bash
curl -X POST \
  "http://API_ENDPOINT/v1/config/knowledgebases/kb9acd1faa893d41b9b5487abe9105b7c7/files/b9b1147c58384e7f891752ea5adc4c35/source" \
  -H "Authorization: EAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_source": "https://aliyuque.antfin.com/pai/arch/ktarnwbn0iqpgy3b"}'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "更新文件来源成功",
  "data": {
    "id": "a0e60dc019a04152b4b884291cee9171",
    "kb_id": "kb9acd1faa893d41b9b5487abe9105b7c7",
    "file_name": "test/pairag.md",
    "file_source": "https://aliyuque.antfin.com/pai/arch/ktarnwbn0iqpgy3b",
    "status": "succeeded",
    "file_metadata": {
      "file_name": "test/pairag.md",
      "file_extension": ".md"
    },
    "updated_at": "2025-08-14T03:36:05.551678"
  }
}
```

> 🔗 在检索结果中可通过 `metadata.file_source` 返回此链接。

---

## 6. 检索知识库内容

在指定知识库中执行语义检索。

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/retrieval`
- **内容类型**：`application/json`

### 请求体（Body）

```json
{
  "knowledge_id": "kb9acd1faa893d41b9b5487abe9105b7c7",
  "query": "RAG",
  "retrieval_setting": {
    "score_threshold": 0.2,
    "top_k": 3
  }
}
```

| 字段名             | 类型     | 必填 | 说明 |
|--------------------|----------|------|------|
| `knowledge_id`     | string   | 是   | 目标知识库 ID |
| `query`            | string   | 是   | 用户查询语句 |
| `retrieval_setting.score_threshold` | number | 否 | 相似度阈值（默认 0.2） |
| `retrieval_setting.top_k`           | integer | 否 | 返回最多前 K 个结果（默认 3） |

### 示例请求

```bash
curl -X POST \
  "http://API_ENDPOINT/v1/retrieval" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_id": "kb9acd1faa893d41b9b5487abe9105b7c7",
    "query": "RAG",
    "retrieval_setting": {
      "score_threshold": 0.2,
      "top_k": 3
    }
  }'
```

### 成功响应（200 OK）

```json
{
  "records": [
    {
      "content": "PAI-RAG 上手教程简介...\n",
      "score": 0.3845832494300215,
      "title": "test/pairag.md",
      "metadata": {
        "file_path": "kb9acd1faa893d41b9b5487abe9105b7c7/docs/test/pairag.md",
        "file_name": "test/pairag.md",
        "file_size": 555,
        "file_extension": ".md",
        "doc_id": "3825a86076d9433c854f1b2cd92445ba",
        "file_source": null,
        "images_info": []
      }
    }
  ]
}
```

| 字段名       | 类型   | 说明 |
|--------------|--------|------|
| `content`    | string | 匹配的文本片段 |
| `score`      | float  | 相似度得分（越大越相关） |
| `title`      | string | 文件名作为标题 |
| `metadata`   | object | 原始文件元数据，包含源链接、路径等 |

---

## 通用响应结构

所有接口返回统一格式：

```json
{
  "code": 200,
  "message": "操作描述",
  "data": { /* 具体数据，可能为对象、数组或 null */ }
}
```

---

## 注意事项

1. 📁 **文件命名唯一性**：同一知识库中不允许存在同名文件，上传同名文件会覆盖旧文件。
2. 🔁 **异步处理机制**：文件上传后需经历解析、向量化等步骤，状态从 `pending` → `succeeded` 可能需要几秒到几分钟。
3. 🔗 **外部链接支持**：通过 `file_source` 可关联语雀、Confluence 等原始文档地址。

