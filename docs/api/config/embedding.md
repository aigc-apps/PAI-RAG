# Embedding 模型管理 API 文档

> 用于管理向量嵌入（Embedding）模型的连接配置，支持 OpenAI 兼容服务与本地模型。

---

## 认证方式

所有请求需携带 Bearer Token：

```http
Authorization: Bearer <your_access_token>
```

---

## 1. 创建 Embedding 模型连接

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/config/embeddings`
- **内容类型**：`application/json`

### 请求体（Body）

| 字段名       | 类型     | 必填 | 说明 |
|--------------|----------|------|------|
| `model_id`   | string   | 是   | 模型唯一标识符，用于前端展示或 API 调用，建议使用模型名称 |
| `endpoint`   | string   | 是   | OpenAI 兼容的 API Endpoint，通常以 `/v1` 结尾（`local` 类型可为空） |
| `api_key`    | string   | 否   | 访问远程服务所需的 API 密钥；`type=local` 时可不传 |
| `type`       | string   | 是   | 模型类型，取值：`openai_like`（远程兼容服务）或 `local`（本地模型） |
| `model_name` | string   | 是   | 实际调用的模型名称，如 `text-embedding-ada-002`、`BAAI/bge-m3` |

> ⚠️ 注意：
> - `endpoint` 和 `api_key` 仅在 `type=openai_like` 时需要。
> - `type=local` 表示从 ModelScope 等平台下载模型至本地运行。

### 示例请求

```bash
curl -X POST 'http://{API_ENDPOINT}/v1/config/embeddings' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "text-embedding-v3",
    "model_name": "text-embedding-v3",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-12345",
    "type": "openai_like"
  }'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "创建embedding模型成功。",
  "data": {
    "id": "c0b58926bfa843e297818f3a55df65dc",
    "model_id": "text-embedding-v3",
    "model_name": "text-embedding-v3",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "type": "openai_like",
    "dimension": null,
    "embed_batch_size": 10,
    "is_ready": true,
    "is_default": false
  }
}
```

| 响应字段           | 类型      | 说明 |
|--------------------|-----------|------|
| `id`               | string    | 系统生成的唯一 UUID |
| `dimension`        | integer?  | 向量维度，若未获取则为 `null` |
| `embed_batch_size` | integer   | 单次嵌入最大文本数，默认为 10 |
| `is_ready`         | boolean   | 模型是否已就绪可用 |
| `is_default`       | boolean   | 是否为系统默认 Embedding 模型 |

---

## 2. 查询 Embedding 模型列表

### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/embeddings`
- **查询参数（可选）**：
  - `page`: 页码（默认 1）
  - `size`: 每页数量（默认 10）

### 示例请求

```bash
curl -X GET 'http://{API_ENDPOINT}/v1/config/embeddings?page=1&size=10' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "获取Embedding模型列表成功",
  "data": {
    "items": [
      {
        "id": "c0b58926bfa843e297818f3a55df65dc",
        "model_id": "text-embedding-v3",
        "model_name": "text-embedding-v3",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "type": "openai_like",
        "dimension": 1536,
        "embed_batch_size": 10,
        "is_ready": true,
        "is_default": false
      },
      {
        "id": "a1d2e3f4g5h6i7j8k9l0m1n2o3p4q5r",
        "model_id": "bge-m3-local",
        "model_name": "BAAI/bge-m3",
        "endpoint": "",
        "type": "local",
        "dimension": 1024,
        "embed_batch_size": 8,
        "is_ready": true,
        "is_default": true
      }
    ],
    "total": 2,
    "pages": 1,
    "page": 1,
    "size": 10
  }
}
```

| 分页字段 | 类型 | 说明 |
|---------|------|------|
| `total` | int  | 总记录数 |
| `pages` | int  | 总页数 |
| `page`  | int  | 当前页码 |
| `size`  | int  | 每页条数 |

---

## 3. 获取单个 Embedding 模型详情

### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/embeddings?model_name={model_name}`  
  > `{model_name}`：模型名称

### 示例请求

```bash
curl -X GET 'http://{API_ENDPOINT}/v1/config/embeddings?model_name=text-embedding-v3' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "获取Embedding模型成功",
  "data": {
    "id": "c0b58926bfa843e297818f3a55df65dc",
    "model_id": "text-embedding-v3",
    "model_name": "text-embedding-v3",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "type": "openai_like",
    "dimension": 1536,
    "embed_batch_size": 10,
    "is_ready": true,
    "is_default": false
  }
}
```

---

## 4. 修改 Embedding 模型连接

> ✅ 支持更新配置，`api_key` 可选更新。

### 请求信息

- **方法**：`PUT`
- **路径**：`/v1/config/embeddings/{id}`
- **内容类型**：`application/json`

> 请求体字段同 [创建接口](#请求体body)，但 `api_key` 可选。若未提供，则保留原有值。

### 示例请求

```bash
curl -X PUT 'http://{API_ENDPOINT}/v1/config/embeddings/c0b58926bfa843e297818f3a55df65dc' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "text-embedding-v3-updated",
    "model_name": "text-embedding-3-large",
    "endpoint": "https://api.openai.com/v1",
    "api_key": "sk-newkeyxxxx",
    "type": "openai_like"
  }'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "Embedding模型更新成功。",
  "data": {
    "id": "c0b58926bfa843e297818f3a55df65dc",
    "model_id": "text-embedding-v3-updated",
    "model_name": "text-embedding-3-large",
    "endpoint": "https://api.openai.com/v1",
    "type": "openai_like",
    "dimension": 3072,
    "embed_batch_size": 10,
    "is_ready": true,
    "is_default": false
  }
}
```

---

## 5. 删除 Embedding 模型连接

> ❗ 删除后无法恢复，请谨慎操作。

### 请求信息

- **方法**：`DELETE`
- **路径**：`/v1/config/embeddings/{id}`

### 示例请求

```bash
curl -X DELETE 'http://{API_ENDPOINT}/v1/config/embeddings/c0b58926bfa843e297818f3a55df65dc' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "Embedding模型ID 'c0b58926bfa843e297818f3a55df65dc' 删除成功。",
  "data": null
}
```

---

## 通用响应结构

所有接口返回统一格式：

```json
{
  "code": 200,
  "message": "操作结果描述",
  "data": { /* 具体数据，可能为对象、数组或 null */ }
}
```

---

## 注意事项

1. ✅ **`type=local` 场景下**：
   - `endpoint` 和 `api_key` 可为空。
   - 模型将从 ModelScope 自动拉取并本地加载。
2. 🔐 **安全性**：`api_key` 不会在查询接口中明文返回。
