# 会话历史管理功能

## 功能概述

自动在 Redis 中保存和恢复用户对话历史，支持多轮对话上下文管理。

## 核心特性

- **自动会话管理**：session_id 为空时自动生成 UUID
- **上下文恢复**：单条消息请求时自动从 Redis 加载历史
- **历史保存**：每轮对话后自动保存到 Redis
- **容量控制**：只保留最近 5 轮对话（10 条消息）
- **自动过期**：7 天后自动清理

## Redis Key 设计

```
session:user:{user_id}:model:{model}:session:{session_id}
```

示例：

```
session:user:alice:model:qwen-max:session:550e8400-e29b-41d4-a716-446655440000
```

## API 使用

### 请求参数

在 `ChatAgentRequest` 中新增了 `session_id` 字段：

```json
{
  "model": "qwen-max",
  "messages": [{ "role": "user", "content": "你好" }],
  "user_id": "alice",
  "session_id": "optional-session-id" // 可选，不传时自动生成
}
```

### 响应头

响应头中会返回 session_id：

```
X-Session-Id: 550e8400-e29b-41d4-a716-446655440000
```

## 使用场景

### 场景 1：新会话

```bash
# 第一条消息（不传 session_id）
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-max",
    "user_id": "alice",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 返回响应头: X-Session-Id: xxx-xxx-xxx
```

### 场景 2：继续会话

```bash
# 第二条消息（传入 session_id 和单条消息）
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-max",
    "user_id": "alice",
    "session_id": "xxx-xxx-xxx",
    "messages": [{"role": "user", "content": "继续聊天"}]
  }'

# 自动从 Redis 加载历史上下文
```

### 场景 3：多轮对话（客户端自己管理上下文）

```bash
# 客户端可以自己维护完整的 messages 数组
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-max",
    "user_id": "alice",
    "session_id": "xxx-xxx-xxx",
    "messages": [
      {"role": "user", "content": "第一条消息"},
      {"role": "assistant", "content": "第一条回复"},
      {"role": "user", "content": "第二条消息"}
    ]
  }'

# 不会从 Redis 加载（messages > 1），但会保存当前轮对话
```

## 工作流程

### 请求流程（chat.py）

1. 检查 `session_id`，为空则生成新的 UUID
2. 将 `session_id` 添加到响应头 `X-Session-Id`
3. 如果 `user_id` 存在且 `messages` 只有 1 条：
   - 从 Redis 加载历史消息
   - 将历史消息添加到 `messages` 前面
4. 保存当前用户消息，用于后续保存

### 响应流程（utils.py）

#### 流式响应（convert_gen_to_stream_chat_completions）

1. 累积每个 chunk 的内容到 `final_content`
2. 流式返回每个 chunk 给客户端
3. 在 `finally` 块中，响应完成后：
   - 检查是否需要保存（`user_id`、`session_id`、`user_message` 都存在）
   - 调用 `session_history_manager.save_messages()` 保存到 Redis

#### 非流式响应（convert_gen_to_chat_completions）

1. 累积完整的助手回复到 `content`
2. 构建完整的 ChatCompletion 对象
3. 返回前检查是否需要保存
4. 调用 `session_history_manager.save_messages()` 保存到 Redis

## 日志示例

```
INFO: Generated new session_id: 550e8400-e29b-41d4-a716-446655440000
INFO: Messages has only one item, attempting to load history from Redis
INFO: Loaded session history: user=alice, session=550e8400-..., messages_count=8
INFO: Restored 8 history messages, total messages: 9
INFO: Saved session history: user=alice, session=550e8400-..., total_messages=10
```

## 配置

在 `SessionHistoryManager` 中可以调整：

```python
MAX_HISTORY_ROUNDS = 5  # 保存最近 5 轮对话
TTL_SECONDS = 7 * 24 * 60 * 60  # 7 天过期
```

## 清除会话历史

如果需要手动清除特定会话的历史：

```python
from service.cache.session_history_manager import session_history_manager

await session_history_manager.clear_history(
    user_id="alice",
    model="qwen-max",
    session_id="550e8400-e29b-41d4-a716-446655440000",
)
```

## 注意事项

1. **user_id 必填**：只有提供 `user_id` 时才会启用会话历史功能
2. **单条消息触发恢复**：只有当 `messages` 数组长度为 1 时才会从 Redis 恢复历史
3. **模型隔离**：不同模型的对话历史是独立的
4. **容错处理**：Redis 操作失败不会影响正常对话流程
5. **流式响应**：支持流式和非流式两种响应模式
