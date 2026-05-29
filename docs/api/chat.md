# Agentic RAG Chat API
## 概述
Agentic RAG Chat API 是一个支持**智能代理（Agent）+ 检索增强生成（RAG）** 的流式聊天接口。它扩展了 OpenAI 兼容的聊天协议，支持知识库检索、多步骤推理、外部工具调用等高级功能，适用于构建智能对话系统、知识问答机器人等场景。

## 核心特性
+ ✅ **流式响应**：基于 `text/event-stream` 实时返回对话内容  
+ 🔍 **RAG 增强**：支持知识库检索（`kb_ids`）和网络搜索（`enable_search`）  
+ 🤖 **智能代理能力**：多步骤推理（`enable_agent`）、MCP 工具调用（`mcp_ids`）  
+ 📎 **附件支持**：处理文件/图片等上下文数据（`enable_attachments`）  
+ ⚙️ **OpenAI 兼容**：返回格式与 OpenAI `chat/completions` 流式响应一致

---

## API 端点
`POST /v1/chat/completions`  
_注：实际路径需根据服务部署配置调整（如 _`/xxx/v1/chat/completions`_）_

---

### 认证方式

所有请求需在请求头中携带认证 Token：

```http
Authorization: EAS_TOKEN
```

> 🔐 请确保 `EAS_TOKEN` 安全存储，不可泄露。

---

### 请求信息

- **方法**：`POST`
- **路径**：`/v1/config/completions`
- **内容类型**：`application/json`

## 请求模式
### chat应用
chat应用模式是基于/v1/chat/completions API 端点设计的简化调用模式，一次性完成一个应用配置后，后续调用该应用进行对话无需重复设置各种参数，旨在降低用户使用门槛，快速发送请求。适用于需要高频调用对话接口、追求参数简化与功能完整性的场景，如客服机器人、智能问答系统等，帮助开发者聚焦业务逻辑，减少接口配置工作量。

建议您首先考虑使用ui界面或者api构建chat应用。使用chat应用进行后续对话。

#### 请求体
```bash
curl -X POST http://api.example.com/v1/chat/completions \
-H "Authorization: Bearer $API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "chatbot",
    "messages": [
         {"role": "user", "content": "杭州在中国哪个省?"}
      ],
    "stream": true,
    
}'
```

#### 请求参数
| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | `string` | ✓ | - | 注册的应用名称（如 `chatbot`） |
| `messages` | `Array` | ✓ | - | 用户输入，格式：`[{"role": "user", "content": "问题"}]` |
| `stream` | `boolean` | ✗ | `true` | **目前默认为 **`true`（本 API 目前仅支持流式响应） |


### 模型调用
如果想要更灵活的调用方式，可以直接通过模型调用。

#### 请求体
```bash
curl -X POST http://api.example.com/v1/chat/completions \
-H "Authorization: $API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-max",
    "messages": [
         {"role": "user", "content": "杭州在中国哪个省?"}
      ],
    "stream": true,
    "mcp_ids": ["4ec7d057aaa64d06b8abe1270da867d4"],
    "kb_ids": ["8edf245bfb8abedadvb472c7"],
    "enable_search": false,
    "enable_agent": false,
    "max_steps": 15,
}'
```



#### 请求参数
| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | `string` | ✓ | - | 模型名称（如 `gpt-4-turbo`, `qwen-max`），仅支持支持function-call的模型 |
| `messages` | `Array` | ✓ | - | 用户输入，格式：`[{"role": "user", "content": "问题"}]` |
| `stream` | `boolean` | ✗ | `true` | **目前默认为 **`true`（本 API 目前仅支持流式响应） |
| `mcp_ids` | `string[]` | ✗ | `[]` | MCP 工具ID（如 `["4ec7d057aaa64d06b8abe1270da867d4"]`） |
| `kb_ids` | `string[]` | ✗ | `[]` | 知识库 ID 列表（如 `["8edf245bfb8abedadvb472c7"]`） |
| `metadata_condition` | `object` | ✗ | `null` | 元数据筛选条件，用于过滤知识库检索范围。支持通过 `conditions`（叶子条件）和 `condition_groups`（嵌套条件组）构建复杂筛选逻辑，最大嵌套深度 5 层。结构说明及示例详见 [检索 API - Metadata Condition](kb_file_management.md#metadata-condition-元数据筛选条件) |
| `enable_search` | `boolean` | ✗ | `false` | 是否启用网络搜索 |
| `enable_agent` | `boolean` | ✗ | `false` | 是否启用多步骤推理（Agentic模式） |
| `max_steps` | `integer` | ✗ | `15` | Agentic模式 多步骤推理最大执行步骤（防无限循环） |


## chat响应对象（流式输出）
  
返回 OpenAI 兼容的流式数据块。示例：  

```json
{
  "id": "chat-xxx",
  "object": "chat.completion.chunk",
  "created": 1717028458,
  "model": "qwen-max",
  "actions": [],
  "observation": "",
  "choices": [{
    "index": 0,
    "delta": {"content": "北京", "role":"assistant","tool_calls":[],"reasoning_content":"","reasoning_completed": False},
    "finish_reason": "stop"
  }],
  "usage": {
       "completion_tokens": 46,
       "prompt_tokens": 2114,
       "total_tokens": 2160,
     },
  "citations": [],
  "citation_details": [],
}
```

| 字段                | 类型        | 说明                                                                 |
| ------------------- | ----------- | -------------------------------------------------------------------- |
| id                  | string      | 本次调用的唯一标识符，每个 chunk 对象的 id 相同。|
| choices             | array       | 模型生成内容的数组，可包含一个或多个 `choices` 对象。|
|                     |             | delta（object）：chat 请求的增量对象。|
|                     |             | - content（string）：chunk 的消息内容。|
|                     |             | - role（string）：增量消息对象的角色，仅在第一个 chunk 中有值。|
|                     |             | - tool_calls（array）：默认空，模型回复要调用的工具及所需参数，可含一个或多个工具调用。|
|                     |             | - reasoning_content（string）：chunk 的 reasoning 消息内容（仅对有思考模式的模型生效）。|
|                     |             | - reasoning_completed（boolean）：reasoning 消息内容是否结束（仅对有思考模式的模型生效，默认 `False`）。|
|                     |             | index（integer）：当前响应在 `choices` 列表中的序列编号，输入参数 `n` 大于 1 时需拼接完整内容。|
|                     |             | finish_reason（string）：因触发输入参数的 `stop` 条件或自然停止输出时为 `stop`。|
| actions             | array       | 要调用的工具及所需参数的 json 表示，可包含一个或多个工具响应对象。|
| observation         | string      | 工具调用结果。|
| created             | integer     | 本次 chat 请求创建时的时间戳，每个 chunk 对象时间戳相同。|
| model               | string      | 本次 chat 请求使用的模型名称。|
| object              | string      | 始终为 `chat.completion.chunk`。|
| usage               | object      | 本次 chat 请求使用的 Token 信息，仅当 `include_usage` 为 `true` 时，在最后一个 chunk 显示。|
|                     |             | - completion_tokens（integer）：模型生成回复转换为 Token 后的长度。|
|                     |             | - prompt_tokens（integer）：用户输入转换为 Token 后的长度。|
|                     |             | - total_tokens（integer）：`prompt_tokens` 与 `completion_tokens` 的总和。|
| citations           | array       | 参考资料的 URL。|
| citation_details    | array       | 参考资料的详细信息。|
|                     |             | - source（string）：来源工具、来源内容。|
|                     |             | - name（string）：来源名称。|
|                     |             | - url（string）：来源 URL。|
|                     |             | - score（float）：分数。|

---

## 图片/视频问答

支持通过上传图片或视频文件，结合多模态大模型（如 Qwen-VL）进行视觉内容问答。

### 前置条件

- 系统中已配置多模态大模型（如 `qwen-vl-plus`、Qwen-VL、Claude 等）
- 所有请求需携带 `X-TENANT-ID` 请求头

---

### 第一步：上传图片文件

**接口**：`POST /v1/files`

**请求示例**：

```bash
curl -s -X POST "http://localhost:8680/v1/files" \
  -H "X-TENANT-ID: your-tenant-id" \
  -F "file=@/path/to/photo.jpg" \
  -F "purpose=chat_attachment"
```

**请求参数**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | `binary` | ✓ | 上传的文件（图片/视频） |
| `purpose` | `string` | ✓ | 用途，图片问答使用 `chat_attachment`（TTL 7天）或 `vision`（TTL 24小时） |
| `metadata` | `string` | ✗ | JSON 格式的自定义元数据 |
| `expires_in` | `integer` | ✗ | 过期时间（秒），0 表示不过期，覆盖 purpose 默认 TTL |

**响应示例**（HTTP 202）：

```json
{
  "data": {
    "id": "file-abc123",
    "tenant_id": "your-tenant-id",
    "purpose": "chat_attachment",
    "file_name": "photo.jpg",
    "file_extension": ".jpg",
    "file_size": 102400,
    "mime_type": "image/jpeg",
    "status": "succeeded",
    "created_at": "2026-05-13T10:00:00",
    "updated_at": "2026-05-13T10:00:00"
  }
}
```

> 📌 **注意**：图片和视频文件上传后会立即标记为 `succeeded` 状态（无需后台处理），请保存返回的 `id` 用于下一步。

---

### 第二步：带图片问答

**接口**：`POST /v1/chat/completions`

**请求示例**：

```bash
curl -sN -X POST "http://localhost:8680/v1/chat/completions" \
  -H "X-TENANT-ID: your-tenant-id" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "这张图片里有什么？",
      "attachments": [
        {"id": "file-abc123", "contentType": "image/jpeg"}
      ]
    }
  ]
}'
```

**请求参数**：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `model` | `string` | ✓ | 模型名称，需支持多模态（如 `qwen-vl-plus`） |
| `stream` | `boolean` | ✗ | 默认 `true`，SSE 流式输出 |
| `messages` | `Array` | ✓ | 消息数组 |
| `messages[].content` | `string` &#124; `Array` | ✓ | 用户提问文本；**仅发送图片不打字时请使用空数组 `[]`**（不要使用空字符串 `""`），系统会自动引导分析 |
| `messages[].attachments` | `Array` | ✓ | 附件数组，引用已上传的文件 |
| `attachments[].id` | `string` | ✓ | 第一步上传返回的文件 ID |
| `attachments[].contentType` | `string` | ✓ | MIME 类型，如 `image/jpeg`、`image/png`、`video/mp4` |

---

### 仅发送图片（无文字）示例

不打字时 `content` 必须是空数组 `[]`，**不要写成空字符串 `""`** —— 否则上游模型可能直接忽略附件，不会触发图片理解工具。

```bash
curl -sN -X POST "http://localhost:8680/v1/chat/completions" \
  -H "X-TENANT-ID: your-tenant-id" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": [],
      "attachments": [
        {"id": "file-abc123", "contentType": "image/jpeg"}
      ]
    }
  ]
}'
```

---

### 多图片示例

```json
{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "请对比这两张图片的区别",
      "attachments": [
        {"id": "file-abc123", "contentType": "image/jpeg"},
        {"id": "file-def456", "contentType": "image/png"}
      ]
    }
  ]
}
```

---

### 视频问答示例

与图片相同流程，仅 `contentType` 改为视频类型：

```json
{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "这个视频讲了什么内容？",
      "attachments": [
        {"id": "file-video789", "contentType": "video/mp4"}
      ]
    }
  ]
}
```

---

### 工作原理

1. 后端检测用户消息中的 `image/*` 或 `video/*` 类型附件
2. 将文件转换为 base64 数据 URI
3. 自动注册 `multimodal-parser` 工具供 Agent 调用
4. Agent 调用多模态大模型（如 Qwen-VL）分析图片/视频内容
5. 如果用户提供了文本问题，作为分析指令传给模型；如果未提供文本，系统会自动引导模型先理解媒体内容再回答

---

### 相关接口

| 接口 | 说明 |
| --- | --- |
| `GET /v1/files/{file_id}` | 查询文件状态 |
| `GET /v1/files/{file_id}/content` | 获取文件原始内容 |
| `GET /v1/files/{file_id}/url` | 获取文件预签名 URL |
| `DELETE /v1/files/{file_id}` | 手动删除文件 |

> 💡 `purpose=chat_attachment` 的文件会在 7 天后自动过期清理。

