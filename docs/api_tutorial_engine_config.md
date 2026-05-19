# 引擎配置校验 API 调用教程

引擎配置校验推荐使用 `/v1/responses`，因为它能暴露结构化工具过程和
HITL 状态。`/v1/chat/completions` 作为通用 OpenAI Chat Completions
入口保留，但只适合只关心最终文本的客户端。旧的 `/v1/runs`、请求体
`session_id`、`conversation_history` 与 `X-Session-Id` 头均已删除。

## 推荐调用

把业务参数一次性写完整，避免 Agent 推断 region、环境、状态或实例：

```bash
BASE_URL=http://127.0.0.1:8000

curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "engine-config-check-embedding-config",
    "input": "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。",
    "stream": true
  }'
```

`response.completed` 只表示本次 Agent run 正常结束，不等同于业务校验通过。
配置存在问题时，最终文本会说明“校验失败”、错误数量和关键错误类型。

## 多轮

多轮推荐使用 `conversation` 或上一轮 response id。

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "conversation": "engine-config-check-embedding-config",
    "input": "再校验同实例的 ranker_config，列出和 embedding_config 的差异",
    "stream": true
  }'
```

或者在上一轮 `response.completed` 中拿到 `id` 后：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "previous_response_id": "resp_xxx",
    "input": "继续上一轮，输出更短摘要",
    "stream": true
  }'
```

## 模型覆盖

`model` 只覆盖当次请求，不修改全局默认模型：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-max",
    "conversation": "engine-config-check-embedding-config",
    "input": "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。",
    "stream": true
  }'
```

## HITL 续答

如果请求显式设置 `allow_hitl:true`，Agent 可能返回 `requires_action`。
客户端用 `previous_response_id` 与 `function_call_output` 续答：

```bash
curl --no-buffer -X POST "$BASE_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "previous_response_id": "resp_pause",
    "input": [{
      "type": "function_call_output",
      "call_id": "call_ask_001",
      "output": "继续校验"
    }],
    "stream": true
  }'
```

## 直接用阿里云 CLI 交叉核对

如果需要绕过 Agent 直接核对配置是否存在，使用 PAI-REC service API：

```bash
aliyun pairecservice list-engine-configs \
  --region cn-beijing \
  --instance-id pairec-cn-inner-khhjd7wnn1geomcirl \
  --environment Prod \
  --status Released \
  --name embedding_config
```

历史文档中的 `aliyun pai DescribeEngineConfig` 不适用于这里的
PAI-REC 引擎配置查询。

## Smoke

```bash
python scripts/api_smoke.py \
  --base-url "$BASE_URL" \
  --query "请校验 PAI-REC 引擎配置：名称 embedding_config，instanceId pairec-cn-inner-khhjd7wnn1geomcirl，region/cluster_id cn-beijing，环境 生产（Prod），status Released。最终只说明是否校验成功、错误数量和关键错误类型，不要输出完整配置或任何凭证字段值。" \
  --conversation engine-config-check-embedding-config \
  --multi-turn
```
