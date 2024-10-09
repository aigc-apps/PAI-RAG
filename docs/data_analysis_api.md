# 流式接口

#### 调用URL

```shell
{EAS_SERVICE_URL}/service/query/data_analysis
# EAS_SERVICE_URL是EAS调用地址
# 如
# http://test-rag-xw2.1730760139076263.cn-hangzhou.pai-eas.aliyuncs.com/service/query/data_analysis
```

#### HTTP Headers

```shell
Authorization: EAS_TOKEN # Eas调用token
Content-Type: application/json
```

#### HTTP Body

```shell
{
  "question": "用户输入",
  "stream": true,
}
```

#### HTTP Response

采用SSE回复，每个chunk都包函is_finished字段和delta字段，其中，

- is_finished表示是否是最后一个包
- delta表示当前输出的token文本

```json
{
  "delta": "数据库",
  "is_finished": false
}
```

当最后一个包到达时，is_finished为true

```json
{
  "delta": "",
  "is_finished": true,
  "session_id": "current_session_id",
  "docs": [
    {
      "text": "SQL执行返回的结果",
      "score": 1.0,
      "metadata": {
        "query_code_instruction": "LLM生成的SQL语句",
        "query_output": "SQL执行返回的结果",
        "col_keys": ["字段关键词"],
        "invalid_flag": "SQL执行成功返回0，SQL执行失败返回1",
        "query_tables": ["SQL查询的目标数据表格"]
      }
    }
  ],
  "new_query": "实际查询的query，多轮对话时可能有帮助"
}
```

#### 调用示例

```shell
curl -X POST \
http://test-rag-xw2.1730760139076263.cn-hangzhou.pai-eas.aliyuncs.com/service/query/data_analysis \
-H "Content-Type: application/json" \
-H "Authorization: YzlmNWQ2YTVjNDVjMTFiM2FjZTQxNmMzOTU1OGU0Mjc2MjNmYmU4OA==" \
-d '{"question":"how many cats", "stream": true}'
```

```json
{"delta": "数据库", "is_finished": false}
{"delta": "查询", "is_finished": false}
{"delta": "结果显示", "is_finished": false}
{"delta": "，", "is_finished": false}
{"delta": "共有", "is_finished": false}
{"delta": "1", "is_finished": false}
{"delta": "只", "is_finished": false}
{"delta": "猫", "is_finished": false}
{"delta": "。", "is_finished": false}
{"delta": "", "is_finished": true, "session_id": "b72a71c059a04238be2289d916231f5d", "docs": [{"text": "[(1,)]", "score": 1.0, "metadata": {"query_code_instruction": "SELECT COUNT(*) FROM pets WHERE PetType = 'cat' limit 100", "query_output": "[(1,)]", "col_keys": ["COUNT(*)"], "invalid_flag": 0, "query_tables": ["pets"]}}], "new_query": "how many cats"}

```

# 非流式接口

非流式接口和流式接口调用方式类似，仅仅在body里的stream字段取值有区别，同时response是一个json对象，包含回复和参考文档信息。

#### 调用URL

```shell
{EAS_SERVICE_URL}/service/query/data_analysis
# EAS_SERVICE_URL是EAS调用地址
# 如
# http://test-rag-xw2.1730760139076263.cn-hangzhou.pai-eas.aliyuncs.com/service/query/data_analysis
```

#### HTTP Headers

```shell
Authorization: EAS_TOKEN # Eas调用token
Content-Type: application/json
```

#### HTTP Body

```shell
{
  "question": "用户输入",
  "stream": false,  # 或者省略
}
```

#### HTTP Response

```json
{
  "answer": "LLM answer",
  "session_id": "current_session_id",
  "docs": [
    {
      "text": "SQL执行返回的结果",
      "score": 1.0,
      "metadata": {
        "query_code_instruction": "LLM生成的SQL语句",
        "query_output": "SQL执行返回的结果",
        "col_keys": ["字段关键词"],
        "invalid_flag": "SQL执行成功返回0，SQL执行失败返回1",
        "query_tables": ["SQL查询的目标数据表格"]
      }
    }
  ],
  "new_query": "实际查询的query，多轮对话时可能有帮助"
}
```

#### 调用示例

```shell
curl -X POST \
http://test-rag-xw2.1730760139076263.cn-hangzhou.pai-eas.aliyuncs.com/service/query/data_analysis \
-H "Content-Type: application/json" \
-H "Authorization: YzlmNWQ2YTVjNDVjMTFiM2FjZTQxNmMzOTU1OGU0Mjc2MjNmYmU4OA==" \
-d '{"question":"how many cats", "stream": false}'
```

```json
{
  "answer": "数据库查询结果显示，共有1只猫。",
  "session_id": "c4191275da4f4b51acaf66bf94b667ab",
  "docs": [
    {
      "text": "[(1,)]",
      "score": 1.0,
      "metadata": {
        "query_code_instruction": "SELECT COUNT(*) FROM pets WHERE PetType = 'cat' limit 100",
        "query_output": "[(1,)]",
        "col_keys": ["COUNT(*)"],
        "invalid_flag": 0,
        "query_tables": ["pets"]
      }
    }
  ],
  "new_query": "how many cats"
}
```
