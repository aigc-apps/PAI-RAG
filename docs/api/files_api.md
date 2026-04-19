# PAI-RAG 文件上传与问答 API

> 本文档描述 `/v1/files` 资源：独立于知识库的文件上传、提取、检索接口，
> 以及与 Agent 聊天流程的协同机制。

---

## 1. 概述

`/v1/files` 是一组独立的顶层资源，用于**对话附件**的全生命周期管理：

- 服务端生成 `file-xxx` 格式的文件 ID
- 按 `(tenant_id, md5, purpose)` 去重
- 按 `purpose` 驱动 TTL（`chat_attachment=7d`、`kb_ingestion=never`、`vision=24h`）
- 引用计数 + 定时 GC 清理过期且无引用的文件
- 异步提取（PDF/DOCX/PPTX/Markdown/Text）+ 分块（≥5KB 文件）
- 支持断点续传（分片上传）+ SSE 状态流

在聊天侧：

- **文本文件**（`.txt`/`.md`/`.pdf`/`.docx`/…）→ Agent 通过 `read-file` 工具读取；大文件走 `search-file-chunks` 按关键字检索
- **图片/视频** → Agent 通过 `multimodal-parser` 工具调用多模态大模型分析
- **空消息 + 附件** → Agent 自动先理解附件再决定后续工具（无需用户显式提问）

---

## 2. 核心概念

### 2.1 File（`pai_file`）

| 字段 | 说明 |
|------|------|
| `id` | `file-<32 位 hex>`，服务端生成 |
| `tenant_id` | 租户隔离键 |
| `purpose` | `chat_attachment` / `kb_ingestion` / `vision` / `avatar` |
| `file_name`, `file_extension`, `file_size`, `file_md5`, `mime_type` | 元数据 |
| `file_path` | 对象存储内部路径（不外暴） |
| `status` | `pending` → `parsing` → `succeeded` / `failed` / `cancelled` |
| `failed_reason` | 失败原因 |
| `ref_count` | 被 message/kb 引用次数，GC 判据 |
| `expires_at` | TTL 到期时间（purpose 驱动，可 `expires_in` 覆盖） |
| `file_metadata` | JSON，含 `truncated_at_extract`、`via=multipart` 等标记 |

Dedup 约束：`UNIQUE (tenant_id, file_md5, purpose)` —— 相同内容同 purpose 重复上传会返回已有记录。

### 2.2 提取文本（`pai_file_text_content`）

对 PDF/DOCX/PPTX/MD/TXT/CSV/XLSX 等可抽取文本的格式，worker 把内容抽出来存这张表。上限 **500KB**，超出设置 `file_metadata.truncated_at_extract = true`。

图片 / 视频 / 其他二进制不抽文本，`status` 上传后立刻标 `succeeded`。

### 2.3 分块（`pai_file_chunk`）

extract 产出的文本超过 **5KB** 时，worker 会把文本按 **500 字符窗口 + 50 字符 overlap** 切成 chunks，用于 `/chunks?query=` 端点和 Agent 的 `search-file-chunks` 工具。

### 2.4 生命周期

```
POST /v1/files
    ↓ 202 (status=pending)
worker 后台：下载 → 抽文本 → [>5KB 时分块] → 标 succeeded
    ↓
消息引用：message.attachments[].id → ref_count++
    ↓
thread 删除：release_attachment_refs → ref_count--
    ↓
GC 扫描（默认每小时）：expires_at < now AND ref_count == 0 → hard_delete
    （级联删 text_content + chunks + object store blob）
```

---

## 3. 认证

所有请求需携带：

```http
X-TENANT-ID: <租户 ID>
```

如启用 `ENABLE_TENANT_ID=true`，缺失会返回 400。

---

## 4. API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/files` | 上传文件 |
| `GET` | `/v1/files/{id}` | 查元数据 / 状态 |
| `GET` | `/v1/files/{id}/content` | 下载原始字节 |
| `GET` | `/v1/files/{id}/text` | 抽取文本（分页） |
| `GET` | `/v1/files/{id}/chunks` | 按关键字检索（大文件） |
| `GET` | `/v1/files/{id}/url` | 预签名 URL |
| `DELETE` | `/v1/files/{id}` | 强删（绕过 ref_count） |
| `GET` | `/v1/files/events` | SSE 状态流 |
| `POST` | `/v1/files/uploads` | 创建分片上传会话 |
| `PUT` | `/v1/files/uploads/{id}/parts/{n}` | 上传分片 |
| `POST` | `/v1/files/uploads/{id}/complete` | 合并分片 |
| `DELETE` | `/v1/files/uploads/{id}` | 取消分片上传 |

### 4.1 上传文件

**`POST /v1/files`** —— `multipart/form-data`

| 字段 | 必填 | 说明 |
|------|------|------|
| `file` | ✅ | 文件二进制 |
| `purpose` | ✅ | `chat_attachment` / `kb_ingestion` / `vision` / `avatar` |
| `expires_in` | ❌ | 秒数，覆盖 purpose 默认 TTL；`0` = 永不过期 |
| `metadata` | ❌ | JSON 字符串，任意业务元数据 |

响应 **202**（立即返回，`status=pending`）：

```json
{
  "code": 200,
  "data": {
    "id": "file-8f7e6d5c4b3a29180716253443526170",
    "tenant_id": "default",
    "purpose": "chat_attachment",
    "file_name": "note.txt",
    "file_extension": ".txt",
    "file_size": 1234,
    "file_md5": "abcd1234…",
    "mime_type": "text/plain",
    "status": "pending",
    "ref_count": 0,
    "expires_at": "2026-04-25T10:00:00Z",
    "created_at": "2026-04-18T10:00:00Z",
    "file_metadata": {}
  }
}
```

**去重命中**：相同 `(tenant, md5, purpose)` 已存在 → 直接返回老记录（`status` 可能已是 `succeeded`），不触发重新抽取。

**Revive**：若老记录是 `failed` / `cancelled` 状态 → 原地复活（重写字节，`status` 归零到 `pending`，重新入队抽取）。

### 4.2 查询文件

**`GET /v1/files/{file_id}`**

返回 `FileRead`（结构同 POST 响应的 `data`）。

### 4.3 下载原始字节

**`GET /v1/files/{file_id}/content`**

返回 `StreamingResponse`，`Content-Type` 为原文件 mime，`Content-Disposition: inline; filename="..."`。

### 4.4 抽取文本（分页）

**`GET /v1/files/{file_id}/text?offset=0&limit=50000`**

| 参数 | 类型 | 默认 | 上限 |
|------|------|------|------|
| `offset` | int | 0 | — |
| `limit` | int | 50000 | 500000 |

```json
{
  "code": 200,
  "data": {
    "file_id": "file-xxx",
    "content": "抽取后的文本片段…",
    "offset": 0,
    "limit": 50000,
    "total_length": 327651,
    "has_more": true,
    "truncated_at_extract": false,
    "extractor_version": "v3"
  }
}
```

- 状态未到 `succeeded` 或文件无可抽文本 → 404
- `truncated_at_extract=true` 表示抽取器本身命中 500KB 上限，分页到 `total_length` 就到头了

### 4.5 按关键字检索

**`GET /v1/files/{file_id}/chunks?query=<关键字>&top_k=5`**

| 参数 | 类型 | 默认 | 范围 |
|------|------|------|------|
| `query` | string | — | 必填 |
| `top_k` | int | 5 | 1–20 |

```json
{
  "code": 200,
  "data": {
    "file_id": "file-xxx",
    "query": "conclusion",
    "total_chunks": 34,
    "hits": [
      {
        "chunk_id": "fchk-...",
        "chunk_index": 27,
        "content": "…段落文本…",
        "start_offset": 13500,
        "end_offset": 14000,
        "score": 3.0
      }
    ]
  }
}
```

- 打分方式：**查询词空格切分后的出现次数之和**（大小写不敏感，无外部依赖）
- 小文件（<5KB）无 chunks，`hits=[]` + `total_chunks=0` → 客户端应回退到 `/text`
- 所有词都未命中时会返回前 `top_k` 个 chunk（score=0）兜底

### 4.6 预签名 URL

**`GET /v1/files/{file_id}/url`**

返回 `{url, expires_at}`，OSS 默认 1 小时，本地存储走 `/v1/fileview` HMAC 签名。

### 4.7 删除

**`DELETE /v1/files/{file_id}`**

强删，**绕过 ref_count**（OpenAI Files 语义）。级联删 text_content、chunks、对象存储 blob。

### 4.8 SSE 状态流

**`GET /v1/files/events?ids=file-a,file-b,file-c`**

订阅一组文件的状态变更。每次 worker 把 status 推进一级就会收到：

```
event: status
data: {"id": "file-a", "status": "parsing"}

event: status
data: {"id": "file-a", "status": "succeeded"}

event: done
data: {}
```

所有文件都到终态（`succeeded` / `failed` / `cancelled`）或 5 分钟超时后自动关闭。

### 4.9 分片上传

适用于单文件 >100MB 或网络不稳定时：

```bash
# 1. 创建会话
POST /v1/files/uploads
    file_name=big.pdf & purpose=chat_attachment & expires_in=86400
    → { upload_id: "upl-xxx", expires_at: "...", parts: [] }

# 2. 逐片上传（part_number 从 1 开始）
PUT /v1/files/uploads/upl-xxx/parts/1
    chunk: <bytes>
PUT /v1/files/uploads/upl-xxx/parts/2
    chunk: <bytes>
    …

# 3. 合并（可选校验分片数）
POST /v1/files/uploads/upl-xxx/complete
    part_count=2
    → FileRead (id, status, …)

# 4. 取消（未 complete 时）
DELETE /v1/files/uploads/upl-xxx
```

合并流程**共享单文件上传的 dedup / revive / 孤儿清理策略**：

- 合并后 md5 在 tenant 内已存在 → 复用，不多写一份 blob
- 老记录是 failed/cancelled → 就地 revive
- `complete` 幂等：会话已是 COMPLETED 时直接返回已关联的 file

---

## 5. 端到端使用示例

环境变量：

```bash
export API=http://localhost:8682
export TENANT=demo
```

### 5.1 上传文本文件并直接读取（小文件路径）

```bash
echo "PAI-RAG is a turnkey RAG system." > /tmp/note.txt

resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/tmp/note.txt" \
  -F "purpose=chat_attachment")
fid=$(echo "$resp" | jq -r '.data.id')

# 等 worker 抽完（PDF 会慢一点）
until [ "$(curl -sH "X-TENANT-ID: $TENANT" "$API/v1/files/$fid" \
  | jq -r '.data.status')" = "succeeded" ]; do sleep 0.5; done

# 读全文
curl -s "$API/v1/files/$fid/text" -H "X-TENANT-ID: $TENANT" | jq '.data'
```

### 5.2 上传 PDF → 检索

```bash
resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/path/to/book.pdf" \
  -F "purpose=chat_attachment")
fid=$(echo "$resp" | jq -r '.data.id')

# 用 SSE 订阅状态（比轮询干净）
curl -N "$API/v1/files/events?ids=$fid" -H "X-TENANT-ID: $TENANT" &
# 当看到 status=succeeded 就 Ctrl-C

# 关键字检索
curl -s "$API/v1/files/$fid/chunks?query=conclusion&top_k=3" \
  -H "X-TENANT-ID: $TENANT" | jq '.data'
```

### 5.3 上传图片 → Agent 识图

```bash
resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/path/to/photo.jpg" \
  -F "purpose=chat_attachment")
img_id=$(echo "$resp" | jq -r '.data.id')

curl -sN -X POST "$API/v1/chat/completions" \
  -H "X-TENANT-ID: $TENANT" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [{
    "role": "user",
    "content": "图里是什么？",
    "attachments": [{"id": "$img_id", "contentType": "image/jpeg"}]
  }]
}
EOF
```

Agent 会自动：
1. 看到 `contentType: image/*` → 路由到 `image_ids`
2. 注册 `multimodal-parser` 工具（base64 data URI 预先拿好）
3. LLM 根据用户 query 决定调 `multimodal-parser`
4. 多模态 LLM 分析图片 → 返回描述
5. 聊天 LLM 把描述编排成最终回答

### 5.4 只上传，不打字（Agent 主动理解）

```bash
# 上传一张菜单照片，不打字
resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/path/to/menu.jpg" \
  -F "purpose=chat_attachment")
img_id=$(echo "$resp" | jq -r '.data.id')

# content 为空字符串 / 空数组都行
curl -sN -X POST "$API/v1/chat/completions" \
  -H "X-TENANT-ID: $TENANT" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "qwen-plus",
  "stream": true,
  "messages": [{
    "role": "user",
    "content": "",
    "attachments": [{"id": "$img_id", "contentType": "image/jpeg"}]
  }]
}
EOF
```

Agent 端行为：
- 检测到"有附件但无文字"
- 给 LLM 注入提示：「用户上传了 1 张图片但没有文字提问。请先调用 `multimodal-parser` 理解附件，识别真实意图；如需额外信息再调搜索 / 知识库等工具。」
- LLM 调 `multimodal-parser` → 拿到"这是一张菜单，包含若干菜品和价格"
- LLM 再根据场景决定：是直接总结、还是顺便搜一下评价 / 营养信息

### 5.5 分片上传 500MB 视频

```bash
FILE=/path/to/video.mp4
SIZE=$(stat -f%z "$FILE")
CHUNK=$((20 * 1024 * 1024))  # 20MB per part

# 1. 创建会话
resp=$(curl -s -X POST "$API/v1/files/uploads" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file_name=video.mp4" \
  -F "purpose=chat_attachment")
upload_id=$(echo "$resp" | jq -r '.data.upload_id')

# 2. 分片上传
part=1
offset=0
while [ $offset -lt $SIZE ]; do
  dd if="$FILE" of=/tmp/chunk bs=1 count=$CHUNK skip=$offset 2>/dev/null
  curl -s -X PUT "$API/v1/files/uploads/$upload_id/parts/$part" \
    -H "X-TENANT-ID: $TENANT" \
    -F "chunk=@/tmp/chunk"
  offset=$((offset + CHUNK))
  part=$((part + 1))
done

# 3. 合并
curl -s -X POST "$API/v1/files/uploads/$upload_id/complete" \
  -H "X-TENANT-ID: $TENANT" -F "part_count=$((part - 1))"
```

---

## 6. Agent 行为速查

| 场景 | 注册的工具 | 触发路径 |
|------|-----------|---------|
| 附件含 `image/*` | `multimodal-parser`（base64） | LLM 按需调用 |
| 附件含 `video/*` | `multimodal-parser`（base64） | LLM 按需调用 |
| 附件含可抽文本 | `read-file`（动态查 DB） | LLM 按需调用 |
| 附件含大文本（>5KB 已分块） | `read-file` + `search-file-chunks` | 先搜再读 |
| 附件含 `.xlsx/.csv/.xls` | 上三条 + codesandbox 工具 | 数据分析场景 |

用户 **无文字仅附件**时，`parse_attachment_tools` 会把"先理解再决定"的显式指令注入 user message，避免 LLM 空转。

`read-file` / `search-file-chunks` 都是**调用时从 DB 读最新数据** + **对 pending 状态轮询到 15s**，避免上传→发送过快时 worker 还没写完内容导致工具返回空。

---

## 7. 配置与限制

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PAIRAG_FILE_GC_INTERVAL_SECONDS` | `3600` | GC 扫描间隔；`0` 禁用 |
| `FILE_STORE_TYPE` | `local` | `local` / `oss` |
| `ENABLE_MINERU` | `0` | PDF OCR（扫描件必需） |
| `DASHSCOPE_API_KEY` | — | 多模态 / 聊天 LLM 凭证 |

### TTL 默认值（可被 `expires_in` 覆盖）

| purpose | TTL |
|---------|-----|
| `chat_attachment` | 7 天 |
| `vision` | 24 小时 |
| `kb_ingestion` | 永不过期 |
| `avatar` | 永不过期 |

### 硬限制

| 项 | 值 | 备注 |
|---|---|---|
| 单次 extract 文本上限 | 500,000 字符 | `truncated_at_extract` 标记溢出 |
| `/text?limit=` | 500,000 | 单次响应上限 |
| Chunk 窗口 | 500 字符 | 重叠 50 |
| Chunking 触发阈值 | 5,000 字符 | 以下不分块 |
| LLM 内联文本 | 5,000 字符 | 超出走 `search-file-chunks` |
| 多片合并在内存 | 几百 MB | 超大文件考虑原生 multipart upload（未实现） |
| 前端单文件上传 | 10MB | Composer 端硬限 |

### 并发安全

- `(tenant, md5, purpose)` 有 UNIQUE 约束 + IntegrityError 兜底：两个请求几乎同时落同一 md5 时，输家回滚 + 删孤儿 blob + 返回赢家
- 分片上传支持重传：PUT 相同 part number 会覆盖
- GC 基于 `ref_count >= n` 谓词的原子 UPDATE，不会把引用计数踩负

---

## 8. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| PDF 上传后一直 `pending` | worker 没起 | `ps aux \| grep celery`；`celery -A app.worker worker --loglevel=info` |
| PDF 抽出来是空的 | 扫描件无文本层 | 设 `ENABLE_MINERU=1` 跑 OCR |
| `/chunks` 返回空 `hits` | 文件 <5KB 不分块 | 回退到 `/text` |
| 图片传了但 Agent 说"没看到" | 未配多模态 LLM | LLMs 页面配一个 `qwen-vl-plus` 并勾多模态 |
| 视频 Agent 不识别 | 参考上条；本地存储 URL 不可达是另一种成因 | 用 OSS 后端或走 base64 data URI |
| 重复上传同文件占空间 | 没命中 dedup | 检查 `purpose` 是否一致 |
| 删 thread 后文件不消失 | 正常：等 TTL + GC | GC 默认每小时扫一次；或 `DELETE /v1/files/{id}` 强删 |

---

## 9. 与旧接口的关系

旧 `/v1/config/attachments*` 端点已**整体移除**。原始行为与新 `/v1/files` 的映射：

| 旧接口 | 新接口 |
|-------|-------|
| `POST /v1/config/attachments`（客户端传 file_id） | `POST /v1/files`（服务端生成 id） |
| `GET /v1/config/attachments/urls?ids=a,b,c` | 多次调 `GET /v1/files/{id}` + `GET /v1/files/{id}/url` |
| 同步等待 5 分钟 | 202 + SSE / 轮询 |
| 塞进 `default_attachments` KB | 独立 `pai_file` 表 |
| 永不清理 | TTL + ref_count + GC |

`message.attachments[].id` 里存的是 `file-xxx`，Agent 的 `parse_attachment_tools` 按 `contentType` 路由到对应工具链。
