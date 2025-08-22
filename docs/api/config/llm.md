# 大模型连接配置 API 文档

> 本文档描述了用于管理大模型（LLM）连接配置的 RESTful API 接口。

---

## 认证方式

所有请求需在请求头中携带 Bearer Token 进行身份验证：

```http
Authorization: Bearer <your_access_token>
```

---

## 1. 创建大模型连接

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/config/llms`
- **内容类型**：`application/json`

### 请求体（Body）

| 字段名            | 类型     | 必填 | 说明 |
|-------------------|----------|------|------|
| `model_id`        | string   | 是   | 模型唯一标识符，用于前端展示或 API 调用，建议使用模型名称；可用于区分同名模型的不同实例 |
| `base_url`        | string   | 是   | OpenAI 兼容的 API Endpoint，通常以 `/v1` 结尾 |
| `api_key`         | string   | 是   | 访问该模型服务所需的 API 密钥 |
| `model`           | string   | 是   | 实际调用的模型名称，例如 `qwen-max`, `Qwen3-8B`, `DeepSeek-R1` |
| `enable_thinking` | boolean  | 否   | 是否为“思考型”模型（如 DeepSeek-R1、Qwen-3 系列），默认为 `false` |
| `vision_support`  | boolean  | 否   | 是否支持多模态输入（图像等），默认为 `false` |
| `temperature`     | number   | 否   | 生成文本的随机性控制参数，取值范围 `[0.0, 1.0]`，默认为 `0.1` |

> ⚠️ 注意：`base_url` 和 `api_key` 应确保正确且可访问，否则可能导致模型调用失败。

### 示例请求

```bash
curl -X POST 'http://{API_ENDPOINT}/v1/config/llms' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "qwen-max",
    "model": "qwen-max",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-123xxx",
    "vision_support": false,
    "enable_thinking": false,
    "temperature": 0.3
  }'
```

### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "LLM创建成功。",
  "data": {
    "id": "f8bf0fcefde44bafa5cc3cf80720f880",
    "model_id": "qwen-max",
    "model": "qwen-max",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "context_window": 8000,
    "temperature": 0.3,
    "enabled": true,
    "vision_support": false,
    "enable_thinking": false,
    "source": "通义千问"
  }
}
```

| 响应字段         | 类型     | 说明 |
|------------------|----------|------|
| `id`             | string   | 系统生成的唯一 UUID，用于后续操作 |
| `context_window` | integer  | 模型上下文窗口大小（token 数） |
| `source`         | string   | 模型来源平台名称（自动识别） |

---

## 2. 查询大模型连接

### 2.1 获取模型列表

#### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/llms`
- **查询参数（可选）**：
  - `page`: 页码（默认 1）
  - `size`: 每页数量（默认 10）

#### 示例请求

```bash
curl -X GET 'http://{API_ENDPOINT}/v1/config/llms?page=1&size=10' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

#### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "获取LLM模型列表成功",
  "data": {
    "items": [
      {
        "id": "24a5af3587a748f1b1adcbb9ad267807",
        "model_id": "qwen-max",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 8000,
        "temperature": 0.3,
        "enabled": true,
        "vision_support": false,
        "enable_thinking": false,
        "source": "通义千问"
      },
      {
        "id": "e71d206914a7412ea2ef0731ea564964",
        "model_id": "qwen-max-vl",
        "model": "qwen-max-vl",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 8000,
        "temperature": 0.3,
        "enabled": true,
        "vision_support": true,
        "enable_thinking": false,
        "source": "通义千问"
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

### 2.2 获取单个模型详情

#### 请求信息

- **方法**：`GET`
- **路径**：`/v1/config/llms/{id}`  
  > `{id}`：创建或查询返回的模型唯一 ID（UUID）

#### 示例请求

```bash
curl -X GET 'http://{API_ENDPOINT}/v1/config/llms/24a5af3587a748f1b1adcbb9ad267807' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

#### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "获取LLM模型成功",
  "data": {
    "id": "24a5af3587a748f1b1adcbb9ad267807",
    "model_id": "qwen-max",
    "model": "qwen-max",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "context_window": 8000,
    "temperature": 0.3,
    "enabled": true,
    "vision_support": false,
    "enable_thinking": false,
    "source": "通义千问"
  }
}
```

---

## 3. 修改大模型连接
#### 请求信息

- **方法**：`PUT`
- **路径**：`/v1/config/llms/{id}`
- **内容类型**：`application/json`

> 请求体字段同 [创建接口](#请求体body)，但 `api_key` 可选。若未提供，则保留原有密钥。

#### 示例请求

```bash
curl -X PUT 'http://{API_ENDPOINT}/v1/config/llms/e71d206914a7412ea2ef0731ea564964' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "qwen-max-vl",
    "model": "qwen-max-vl",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-newkey123", 
    "vision_support": true,
    "enable_thinking": false,
    "temperature": 0.5
  }'
```

#### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "LLM更新成功。",
  "data": {
    "id": "e71d206914a7412ea2ef0731ea564964",
    "model_id": "qwen-max-vl",
    "model": "qwen-max-vl",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "context_window": 8000,
    "temperature": 0.5,
    "enabled": true,
    "vision_support": true,
    "enable_thinking": false,
    "source": "通义千问"
  }
}
```

---

## 4. 删除大模型连接

#### 请求信息

- **方法**：`DELETE`
- **路径**：`/v1/config/llms/{id}`

#### 示例请求

```bash
curl -X DELETE 'http://{API_ENDPOINT}/v1/config/llms/24a5af3587a748f1b1adcbb9ad267807' \
  -H 'Authorization: Bearer YOUR_BEARER_TOKEN'
```

#### 成功响应（200 OK）

```json
{
  "code": 200,
  "message": "大模型ID '24a5af3587a748f1b1adcbb9ad267807' 删除成功。",
  "data": null
}
```

> 删除后该模型无法再被调用，请谨慎操作。

---

## 通用响应结构

所有接口返回统一格式：

```json
{
  "code": 200,
  "message": "操作描述信息",
  "data": { /* 返回的具体数据，可能为对象、数组或 null */ }
}
```