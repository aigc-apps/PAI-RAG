# API Service

您可以通过API调用PAI-RAG服务端，以下介绍常用的接口类型和调用方式

# 1. Chat API

## 1.1 OpenAI-Compatiable API

包含功能：

- web search：联网搜索
- chat knowledgebase: 知识库查询
- chat llm: 与LLM聊天
- chat agent: 智能体工具调用
- chat db: 数据库/表格查询

调用方式

- 调用地址：{EAS_SERVICE_URL}/v1/chat/completions
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- HTTP Body 样例如下：
  ```python
  {
      "model": "default",  # 模型名称，填default
      "messages": [
          {"role": "user", "content": "你好"},
          {"role": "assistant", "content": "你好，有什么能帮到您？"},
          {"role": "user", "content": "浙江省会是哪里"},
          {"role": "assistant", "content": "杭州是浙江的省会。"},
          {"role": "user", "content": "有哪些好玩的"},
      ],
      "stream": true,  # 是否流式
      "chat_knowledgebase": true,  # 是否查询本地知识库
      "search_web": false,  # 是否使用联网搜索
      "chat_llm": false,  # 是否仅使用llm聊天
      "chat_agent": false,  # 是否使用agent
      "chat_db": false,  # 是否查询数据库
      "return_reference": false,  # 是否返回参考
      "index_name": "default",  # 索引名称，RAG场景使用，不传使用默认索引
  }
  ```
  **注意：**
- 如果所有功能开关有多个true，会按照以下优先级规则进行调用：
  search_web > chat_knowledgebase > chat_agent > chat_db > chat_llm，在每个功能中，会有前置意图识别区分是否调用该功能或直接llm回复
- 如果所有功能都为false或者都不传，则默认查询本地知识库，chat_knowledgebase=true

<details>
<summary>联网搜索示例</summary>

```python
from openai import OpenAI

##### API 配置 #####
openai_api_key = "EAS_TOKEN"
openai_api_base = "EAS_URL/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


#### Chat ######
def chat():
    stream = True
    chat_completion = client.chat.completions.create(
        model="default",
        stream=stream,
        messages=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么能帮到您？"},
            {"role": "user", "content": "浙江省会是哪里"},
            {"role": "assistant", "content": "杭州是浙江的省会。"},
            {"role": "user", "content": "有哪些好玩的"},
        ],
        extra_body={
            "search_web": True,
        },
    )

    if stream:
        for chunk in chat_completion:
            print(chunk.choices[0].delta.content, end="")
    else:
        result = chat_completion.choices[0].message.content
        print(result)


chat()
```

- 返回 - 非流式输出

  ```json
  {
    "id": "8df97a998f29485fb6b8f0fa1d65c52c",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": null,
        "message": {
          "content": "杭州有很多好玩的地方，以下是一些推荐的景点：\n\n1. **西湖** - 首批国家5A级旅游景区，中国十大风景名胜之一，以自然与人文景观著称。\n2. **西溪国家湿地公园** - 适合亲近自然，感受湿地的魅力。\n3. **灵隐寺** - 杭州最古老的寺庙之一，具有深厚的文化底蕴。\n4. **六和塔** - 杭州的标志性建筑之一，可以登塔俯瞰江景。\n5. **宋城** - 可以体验宋朝的历史文化。\n6. **雷峰塔** - “雷峰夕照”是杭州一大美景，可以欣赏夕阳下的西湖。\n7. **湘湖** - 湖光山色、古桥流水，适合休闲活动。\n8. **钱塘江大桥** - 不仅实用，还是观赏钱塘江壮丽景色的好地方。\n9. **京杭大运河** - 体验古代水运文化的绝佳地点。\n10. **太子湾公园** - 自然与人文景观结合的美丽公园。\n\n此外，还有其他一些免费景点也非常值得一去，比如杭州植物园、法喜寺、胡雪岩故居等。",
          "refusal": null,
          "role": "assistant",
          "audio": null,
          "function_call": null,
          "tool_calls": null
        }
      }
    ],
    "created": 1739450868,
    "model": "qwen-turbo",
    "object": "chat.completion",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
      "completion_tokens": 0,
      "prompt_tokens": 0,
      "total_tokens": 0,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    }
  }
  ```

- 返回 - 流式输出（SSE格式）, chunk 结构如下：
  ```json
  {
    "id": "7eb65e8cbc62428ca7ae22782addc4d1",
    "choices": [
      {
        "delta": {
          "content": "坊",
          "function_call": null,
          "refusal": null,
          "role": "assistant",
          "tool_calls": null
        },
        "finish_reason": null,
        "index": 240,
        "logprobs": null
      }
    ],
    "created": 1739451105,
    "model": "DeepSeek-R1-Distill-Qwen-32B",
    "object": "chat.completion.chunk",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": null
  }
  ```

</details>

<details>
<summary>查询数据库示例</summary>

```python
from openai import OpenAI

##### API 配置 #####
openai_api_key = "EAS_TOKEN"
openai_api_base = "EAS_URL/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


#### Chat ######
def chat():
    stream = True
    chat_completion = client.chat.completions.create(
        model="default",
        stream=stream,
        messages=[
            {"role": "user", "content": "有多少只猫"},
            {"role": "assistant", "content": "有2只猫"},
            {"role": "user", "content": "狗呢"},
        ],
        extra_body={
            "chat_db": True,
        },
    )

    if stream:
        for chunk in chat_completion:
            print(chunk.choices[0].delta.content, end="")
    else:
        result = chat_completion.choices[0].message.content
        print(result)


chat()
```

- 返回 - 非流式输出

  ```json
  {
    "id": "4e35b5366d3a4719b4c5dcdf18cf5769",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": null,
        "message": {
          "content": "有2条狗。",
          "refusal": null,
          "role": "assistant",
          "audio": null,
          "function_call": null,
          "tool_calls": null
        }
      }
    ],
    "created": 1740995847,
    "model": "qwen-max",
    "object": "chat.completion",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
      "completion_tokens": 5,
      "prompt_tokens": 332,
      "total_tokens": 337,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    },
    "citations": [null],
    "citation_details": [
      {
        "name": "SQL Information",
        "text": "{\"SQL\": \"SELECT count(*) FROM pets WHERE PetType = 'dog'\", \"SQL_Exec_Result\": \"[(2,)]\", \"Tables\": [\"pets\"], \"Valid\": 0}",
        "url": null,
        "score": 1.0
      }
    ]
  }
  ```

- 返回 - 流式输出（SSE格式）
  - chunk 结构（初始和中间）：
  ```json
  {
    "id": "b3bbe8da27b648288d96a55718b18f7c",
    "choices": [
      {
        "delta": {
          "content": "有",
          "function_call": null,
          "refusal": null,
          "role": "assistant",
          "tool_calls": null
        },
        "finish_reason": null,
        "index": 0,
        "logprobs": null
      }
    ],
    "created": 1740729820,
    "model": "qwen-max",
    "object": "chat.completion.chunk",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
      "completion_tokens": 0,
      "prompt_tokens": 0,
      "total_tokens": 0,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    },
    "citations": [],
    "citation_details": []
  }
  ```
  - chunk 结构（结尾）
  ```json
  {
    "id": "b3bbe8da27b648288d96a55718b18f7c",
    "choices": [
      {
        "delta": {
          "content": "",
          "function_call": null,
          "refusal": null,
          "role": "assistant",
          "tool_calls": null
        },
        "finish_reason": "stop",
        "index": 2,
        "logprobs": null
      }
    ],
    "created": 1740729820,
    "model": "qwen-max",
    "object": "chat.completion.chunk",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
      "completion_tokens": 5,
      "prompt_tokens": 541,
      "total_tokens": 546,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    },
    "citations": [null],
    "citation_details": [
      {
        "name": "SQL Information",
        "text": "{\"SQL\": \"SELECT count(*) FROM pets WHERE PetType = 'dog'\", \"SQL_Exec_Result\": \"[(2,)]\", \"Tables\": [\"pets\"], \"Valid\": 0}",
        "url": null,
        "score": 1.0
      }
    ]
  }
  ```

</details>

# 2. Knowledgebase API

### 知识库上传接口

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}/files
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: multipart/form-data
- 请求参数：

  - files: 文件
  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/knowledgebases/my_milvus/files \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@example_data/paul_graham/paul_graham_essay.txt'
  ```

- 如果需要上传多份文档，可以使用多个 -F 'files=@path' 参数，每个参数对应一个要上传的文件，示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/knowledgebases/my_milvus/files \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@example_data/paul_graham/paul_graham_essay.txt' \
  -F 'files=@example_data/another_file1.md' \
  -F 'files=@example_data/another_file2.pdf' \
  -F 'index_name=default'
  ```

- 返回示例：
  ```json
  { "message": "Files have been successfully uploaded." }
  ```

### 知识库查询文件上传状态

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1//knowledgebases/{name}/files/{file_name}
- 请求方式：GET
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - file_name: 文件名称
  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/my_milvus/files/paul_graham_essay.txt -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  {
    "task_id": "50fe181921a83edf63b5ecaa487ec61e",
    "operation": "UPDATE",
    "file_name": "localdata/knowledgebase/my_milvus/docs/paul_graham_essay.txt",
    "status": "done",
    "message": null,
    "last_modified_time": "2025-03-12 11:50:37"
  }
  ```

### 知识库查询上传历史

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1//knowledgebases/{name}/history
- 请求方式：GET
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - file_name: 文件名称
  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/my_milvus/history -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  [
    {
      "task_id": "50fe181921a83edf63b5ecaa487ec61e",
      "operation": "UPDATE",
      "file_name": "localdata/knowledgebase/my_milvus/docs/paul_graham_essay.txt",
      "status": "done",
      "message": null,
      "last_modified_time": "2025-03-12 11:50:37"
    },
    {
      "task_id": "0162e61cbe605ddab865fff5f7d8b5e1",
      "operation": "ADD",
      "file_name": "localdata/knowledgebase/my_milvus/docs/三国演义_demo.pdf",
      "status": "failed",
      "message": "Error loading file",
      "last_modified_time": "2025-03-12 13:46:11"
    }
  ]
  ```

### 知识库删除文件接口

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}/files/{file_name}
- 请求方式：DELETE
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - file_name: 文件名称
  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'DELETE' http://localhost:8680/api/v1/knowledgebases/my_milvus/files/paul_graham_essay.txt -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  { "message": "File 'paul_graham_essay.txt' have been successfully removed." }
  ```

其他知识库管理类接口

### 查询指定知识库信息

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}
- 请求方式：GET
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/my_milvus -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  {
    "name": "my_milvus",
    "vector_store_config": {
      "persist_path": "./localdata/knowledgebase/default/.index/.faiss",
      "type": "milvus",
      "is_image_store": false,
      "host": "c-exxx11f4c.milvus.aliyuncs.com",
      "port": 19530,
      "user": "root",
      "password": "xxx",
      "database": "default",
      "collection_name": "test",
      "reranker_weights": [0.5, 0.5]
    },
    "embedding_config": {
      "source": "huggingface",
      "model": "bge-m3",
      "embed_batch_size": 10,
      "enable_sparse": false
    },
    "knowledgebase_paths": {
      "base_path": "localdata/knowledgebase/my_milvus",
      "docs_path": "localdata/knowledgebase/my_milvus/docs",
      "index_path": "localdata/knowledgebase/my_milvus/.index",
      "logs_path": "localdata/knowledgebase/my_milvus/.logs",
      "parse_path": "localdata/knowledgebase/my_milvus/.index/parse",
      "split_path": "localdata/knowledgebase/my_milvus/.index/split",
      "embed_path": "localdata/knowledgebase/my_milvus/.index/embed",
      "faiss_index_path": "localdata/knowledgebase/my_milvus/.index/.faiss",
      "doc_ids_map_file": "localdata/knowledgebase/my_milvus/.index/file_to_docid_map.json"
    }
  }
  ```

### 新增知识库

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: application/json
- 请求参数：

  - name: 新知识库名称，假设叫new_milvus
  - knowledgebase: 新知识库配置

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/knowledgebases/new_milvus \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
        "name":"new_milvus",
        "vector_store_config":
        {
            "persist_path":"./localdata/knowledgebase/default/.index/.faiss",
            "type":"milvus",
            "is_image_store":false,
            "host":"c-exxx11f4c.milvus.aliyuncs.com",
            "port":19530,
            "user":"root",
            "password":"xxx",
            "database":"default",
            "collection_name":"test",
            "reranker_weights":[0.5,0.5]
        },
        "embedding_config":
        {
            "source":"huggingface",
            "model":"bge-m3",
            "embed_batch_size":10,
            "enable_sparse":false
        }
    }'
  ```

- 返回示例：
  ```json
  { "msg": "Add knowledgebase 'new_milvus' successfully." }
  ```

### 更新指定知识库

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}
- 请求方式：PATCH
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: application/json
- 请求参数：

  - name: 知识库名称，假设叫new_milvus
  - knowledgebase: 知识库配置

- curl 请求示例：

  ```bash
  curl -X 'PATCH' http://localhost:8680/api/v1/knowledgebases/new_milvus \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
        "name":"new_milvus",
        "vector_store_config":
        {
            "persist_path":"./localdata/knowledgebase/default/.index/.faiss",
            "type":"milvus",
            "is_image_store":true,
            "host":"c-exxx11f4c.milvus.aliyuncs.com",
            "port":19530,
            "user":"root",
            "password":"xxx",
            "database":"default",
            "collection_name":"test",
            "reranker_weights":[0.5,0.5]
        },
        "embedding_config":
        {
            "source":"huggingface",
            "model":"bge-m3",
            "embed_batch_size":10,
            "enable_sparse":false
        }
    }'
  ```

- 返回示例：
  ```json
  { "msg": "Update knowledgebase 'new_milvus' successfully." }
  ```

### 删除指定知识库

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}
- 请求方式：DELETE
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - name: 知识库名称，假设叫new_milvus

- curl 请求示例：

  ```bash
  curl -X 'DELETE' http://localhost:8680/api/v1/knowledgebases/new_milvus -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  { "msg": "Delete knowledgebase 'new_milvus' successfully." }
  ```

### 知识库列表

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases
- 请求方式：GET
- 请求 HEADERS

  - Authorization: EAS_TOKEN # Eas调用token

- curl 请求示例：

  ```bash
  curl -X 'GET' http://localhost:8680/api/v1/knowledgebases -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  {
    "knowledgebases": {
      "default": {
        "name": "default",
        "vector_store_config": {
          "persist_path": "./localdata/knowledgebase/default/.index/.faiss",
          "type": "milvus",
          "is_image_store": false,
          "host": "c-exxx1911f4c.milvus.aliyuncs.com",
          "port": 19530,
          "user": "root",
          "password": "xxx",
          "database": "default",
          "collection_name": "test",
          "reranker_weights": [0.5, 0.5]
        },
        "embedding_config": {
          "source": "huggingface",
          "model": "bge-m3",
          "embed_batch_size": 10,
          "enable_sparse": false
        },
        "knowledgebase_paths": {
          "base_path": "localdata/knowledgebase/default",
          "docs_path": "localdata/knowledgebase/default/docs",
          "index_path": "localdata/knowledgebase/default/.index",
          "logs_path": "localdata/knowledgebase/default/.logs",
          "parse_path": "localdata/knowledgebase/default/.index/parse",
          "split_path": "localdata/knowledgebase/default/.index/split",
          "embed_path": "localdata/knowledgebase/default/.index/embed",
          "faiss_index_path": "localdata/knowledgebase/default/.index/.faiss",
          "doc_ids_map_file": "localdata/knowledgebase/default/.index/file_to_docid_map.json"
        }
      },
      "my_milvus": {
        "name": "my_milvus",
        "vector_store_config": {
          "persist_path": "./localdata/knowledgebase/default/.index/.faiss",
          "type": "milvus",
          "is_image_store": false,
          "host": "c-e6xxx11f4c.milvus.aliyuncs.com",
          "port": 19530,
          "user": "root",
          "password": "xxx",
          "database": "default",
          "collection_name": "test",
          "reranker_weights": [0.5, 0.5]
        },
        "embedding_config": {
          "source": "huggingface",
          "model": "bge-m3",
          "embed_batch_size": 10,
          "enable_sparse": false
        },
        "knowledgebase_paths": {
          "base_path": "localdata/knowledgebase/my_milvus",
          "docs_path": "localdata/knowledgebase/my_milvus/docs",
          "index_path": "localdata/knowledgebase/my_milvus/.index",
          "logs_path": "localdata/knowledgebase/my_milvus/.logs",
          "parse_path": "localdata/knowledgebase/my_milvus/.index/parse",
          "split_path": "localdata/knowledgebase/my_milvus/.index/split",
          "embed_path": "localdata/knowledgebase/my_milvus/.index/embed",
          "faiss_index_path": "localdata/knowledgebase/my_milvus/.index/.faiss",
          "doc_ids_map_file": "localdata/knowledgebase/my_milvus/.index/file_to_docid_map.json"
        }
      }
    }
  }
  ```

### 知识库文件列表

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/knowledgebases/{name}/files
- 请求方式：GET
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：

  - name: 知识库名称，假设叫my_milvus

- curl 请求示例：

  ```bash
  curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/my_milvus/files -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  [
    {
      "file_name": "localdata/knowledgebase/my_milvus/docs/paul_graham_essay.txt",
      "doc_id": "50fe181921a83edf63b5ecaa487ec61e",
      "last_modified_time": "2025-03-12 14:38:51"
    }
  ]
  ```

**注意：** 旧版知识库管理API请参考[API Service v0.2.0](./api_v0.2.0.md)。

# 3. Other API

## 3.1 RAG配置服务

### 获取RAG配置

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/config
- 请求方式：GET
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'GET' '{EAS_SERVICE_URL}/api/v1/config' -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  {
    "system": {
      "query_type": "websearch"
    },
    "data_reader": {
      "concat_csv_rows": false,
      "enable_mandatory_ocr": false,
      "format_sheet_data_to_json": false,
      "sheet_column_filters": null,
      "number_workers": 4
    },
    "node_parser": {
      "type": "Sentence",
      "chunk_size": 500,
      "chunk_overlap": 10,
      "enable_multimodal": true,
      "paragraph_separator": "\n\n\n",
      "sentence_window_size": 3,
      "sentence_chunk_overlap": 200,
      "breakpoint_percentile_threshold": 95,
      "buffer_size": 1
    },
    "index": {
      "vector_store": {
        "persist_path": "./localdata/knowledgebase/default/.index/.faiss",
        "type": "faiss",
        "is_image_store": false
      },
      "enable_multimodal": true,
      "persist_path": "localdata/storage"
    },
    "embedding": {
      "source": "huggingface",
      "model": "bge-m3",
      "embed_batch_size": 10,
      "enable_sparse": false
    },
    "multimodal_embedding": {
      "source": "cnclip",
      "model": "ViT-L-14",
      "embed_batch_size": 10,
      "enable_sparse": false
    },
    "llm": {
      "source": "openai_compatible",
      "temperature": 0.1,
      "system_prompt": null,
      "max_tokens": 4000,
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key": "sk-xxx",
      "model": "qwen-max"
    },
    "multimodal_llm": {
      "source": "openai_compatible",
      "temperature": 0.1,
      "system_prompt": null,
      "max_tokens": 4000,
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key": "sk-xxx",
      "model": ""
    },
    "functioncalling_llm": null,
    "agent": {
      "system_prompt": "你是一个旅游小助手，xxx",
      "python_scripts": "xxx",
      "function_definition": "xxx",
      "api_definition": "xxx"
    },
    "chat_store": {
      "type": "local",
      "persist_path": "localdata/storage"
    },
    "data_analysis": {
      "type": "mysql",
      "nl2sql_prompt": "给定一个输入问题，xxx",
      "synthesizer_prompt": "给定一个输入问题，xxx",
      "database": "my_pets",
      "tables": [],
      "descriptions": {},
      "enable_enhanced_description": false,
      "enable_db_history": true,
      "enable_db_embedding": true,
      "max_col_num": 100,
      "max_val_num": 1000,
      "enable_query_preprocessor": true,
      "enable_db_preretriever": true,
      "enable_db_selector": true,
      "user": "root",
      "password": "xxx",
      "host": "127.0.0.1",
      "port": 3306
    },
    "intent": {
      "descriptions": {
        "rag": "\nThis tool can help you get more specific information from the knowledge base.\n",
        "tool": "\nThis tool can help you get travel information about time, weather, flights, train and hotels.\n"
      }
    },
    "node_enhancement": {
      "tree_depth": 3,
      "max_clusters": 52,
      "proba_threshold": 0.1
    },
    "oss_store": {
      "bucket": "",
      "endpoint": "oss-cn-hangzhou.aliyuncs.com",
      "ak": null,
      "sk": null
    },
    "postprocessor": {
      "reranker_type": "no-reranker",
      "similarity_threshold": 0.5
    },
    "retriever": {
      "vector_store_query_mode": "default",
      "similarity_top_k": 3,
      "image_similarity_top_k": 2,
      "search_image": false,
      "hybrid_fusion_weights": [0.7, 0.3]
    },
    "search": {
      "source": "google",
      "search_count": 10,
      "serpapi_key": "142xxx",
      "search_lang": "zh-CN"
    },
    "synthesizer": {
      "use_multimodal_llm": false,
      "system_role_template": "你是xxx",
      "custom_prompt_template": "你的目标是提供准确、有用且易于理解的信息。xxx"
    },
    "query_rewrite": {
      "enabled": true,
      "rewrite_prompt_template": "# 角色\n你是一位专业的信息检索专家，xxx",
      "llm": {
        "source": "openai_compatible",
        "temperature": 0.1,
        "system_prompt": null,
        "max_tokens": 4000,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": null,
        "model": ""
      }
    },
    "guardrail": {
      "endpoint": null,
      "region": null,
      "access_key_id": null,
      "access_key_secret": null,
      "custom_advice": null
    }
  }
  ```
  </details>

### 更新RAG配置

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/config
- 请求方式：PATCH
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: application/json
- 请求参数：
  - new_config: 更新的配置信息

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
    curl -X 'PATCH' '{EAS_SERVICE_URL}/api/v1/config' \
    -H 'Authorization: EAS_TOKEN' \
    -H 'Content-Type: application/json' \
    -d '{
        "system": {
          "query_type": "websearch"
        },
        "data_reader": {
          "concat_csv_rows": false,
          "enable_mandatory_ocr": false,
          "format_sheet_data_to_json": false,
          "sheet_column_filters": null,
          "number_workers": 4
        },
        "node_parser": {
          "type": "Sentence",
          "chunk_size": 500,
          "chunk_overlap": 10,
          "enable_multimodal": true,
          "paragraph_separator": "\n\n\n",
          "sentence_window_size": 3,
          "sentence_chunk_overlap": 200,
          "breakpoint_percentile_threshold": 95,
          "buffer_size": 1
        },
        ...
    }' #(更多配置信息可参考 获取RAG配置 的返回示例)
  ```

- 返回示例：
  ```json
  { "msg": "Update RAG configuration successfully." }
  ```

</details>

## 3.2 CHAT_DB信息加载

### 上传excel/csv文件用于chat_db的表格内容查询

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/upload_datasheet
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: multipart/form-data
- 请求参数：
  - file: excel/csv文件

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/upload_datasheet \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@example_data/titanic_train.csv'
  ```

- 返回示例：
  ```json
  {
    "task_id": "3b12cf5fabee4a99a32895d2f6935c0d",
    "destination_path": "./localdata/data_analysis/titanic_train.csv",
    "data_preview": "xxx"
  }
  ```

</details>

### 上传json文件用于chat_db的数据库信息补充——问答对

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/upload_db_history
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: multipart/form-data
- 请求参数：
  - file: json文件
  - db_name: 数据库名称

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/upload_db_history \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@example_data/db_query_history.json' \
  -F 'db_name=my_pets'
  ```

- 返回示例：
  ```json
  {
    "task_id": "204191f946384a54a48b13ec00fd5374",
    "destination_path": "./localdata/data_analysis/text2sql/history/my_pets_db_query_history.json"
  }
  ```

</details>

### 上传csv文件用于chat_db的数据库信息补充——列描述

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/upload_db_history
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
  - Content-Type: multipart/form-data
- 请求参数：
  - files: csv文件
  - db_name: 数据库名称

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/upload_db_description \
  -H 'Authorization: EAS_TOKEN' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@example_data/database_description/schools.csv' \
  -F 'db_name=california_schools'
  ```

- 返回示例：
  ```json
  {
    "task_id": "f417e436cf8b4c329f7b48a7f3c4af64",
    "destination_path": "./localdata/data_analysis/text2sql/input_description"
  }
  ```

</details>

### 加载数据库信息

调用方式

- 调用地址：{EAS_SERVICE_URL}/api/v1/query/load_db_info
- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN # Eas调用token
- 请求参数：无

<details>
<summary>调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' http://localhost:8680/api/v1/load_db_info -H 'Authorization: EAS_TOKEN'
  ```

- 返回示例：
  ```json
  { "task_id": "2389f546af2b6c359d7b19c8b5c3bf88" }
  ```

</details>

# 4. Legacy API （deprecated）

您也可以调用以下API实现单独功能

- **知识库查询：** {EAS_SERVICE_URL}/api/v1/query
- **与大语言模型聊天：** {EAS_SERVICE_URL}/api/v1/query/llm
- **知识库检索：** {EAS_SERVICE_URL}/api/v1/query/retrieval
- **智能体：** {EAS_SERVICE_URL}/api/v1/query/agent
- **网络搜索：** {EAS_SERVICE_URL}/api/v1/query/llm
- **数据分析：** {EAS_SERVICE_URL}/api/v1/query/data_analysis

通用的请求方式、请求参数、以及部分示例如下：

- 请求方式：POST
- 请求 HEADERS
  - Authorization: EAS_TOKEN
  - Content-Type: application/json
- 请求参数：
  - messages: 上下文对话
  - stream: 是否流式输出
  - citation: 是否使用引用标签
  - index_name: 索引名称
  - with_intent: 是否使用意图
  - search_web: 是否搜索网络
  - return_reference: 是否返回参考文档

<details>
<summary>知识库查询调用示例</summary>

- curl 请求示例：

  ```bash
    curl -X 'POST' '{EAS_SERVICE_URL}/api/v1/query' \
    -H 'Authorization: EAS_TOKEN' \
    -H 'Content-Type: application/json' \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": "What programs the author write?"
            }
        ],
        "stream": false,
        "index_name": "default",
        "return_reference": true
    }'
  ```

- 返回示例：
  ```json
  {
    "answer": "The author's first programs were written on an IBM 1401, using an early version of Fortran. These programs had to be typed onto punch cards, which were then fed into the card reader to load and run the program. The output was usually printed on a loud printer. Since the only form of input for these programs was data stored on punched cards, and the author didn't have much data, the programs couldn't do very much. They were likely simple tasks, such as calculating approximations of pi, but the author mentions not being able to recall any specific programs because they weren't very complex or memorable. One clear memory the author has is learning that programs could fail to terminate, after one of his own programs did not stop running, which caused some trouble in the data center.",
    "session_id": "0c6a5a5b7a8743c0b88243cd2934b0f6",
    "docs": [
      {
        "text": "What I Worked On\n\nFebruary 2021\n\nBefore college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories...",
        "score": 0.874595,
        "metadata": {
          "file_path": "50fe181921a83edf63b5ecaa487ec61e/paul_graham_essay.txt",
          "file_name": "paul_graham_essay.txt",
          "file_type": "text/plain",
          "file_size": 7796,
          "creation_date": "2025-03-06",
          "last_modified_date": "2025-03-06"
        },
        "image_url": null
      },
      {
        "text": "With microcomputers, everything changed. Now you could have a computer sitting right in front of you, on a desk, that could respond to your keystrokes as it was running instead of just churning through a stack of punch cards and then stopping...",
        "score": 0.7700965595088596,
        "metadata": {
          "file_path": "50fe181921a83edf63b5ecaa487ec61e/paul_graham_essay.txt",
          "file_name": "paul_graham_essay.txt",
          "file_type": "text/plain",
          "file_size": 7796,
          "creation_date": "2025-03-06",
          "last_modified_date": "2025-03-06"
        },
        "image_url": null
      },
      {
        "text": "So I decided to switch to AI.\n\nAI was in the air in the mid 1980s, but there were two things especially that made me want to work on it: a novel by Heinlein called The Moon is a Harsh Mistress, which featured an intelligent computer called Mike, and a PBS documentary that showed Terry Winograd using SHRDLU...",
        "score": 0.5376133,
        "metadata": {
          "file_path": "50fe181921a83edf63b5ecaa487ec61e/paul_graham_essay.txt",
          "file_name": "paul_graham_essay.txt",
          "file_type": "text/plain",
          "file_size": 7796,
          "creation_date": "2025-03-06",
          "last_modified_date": "2025-03-06"
        },
        "image_url": null
      }
    ],
    "new_query": "first programs the author wrote"
  }
  ```

</details>

<details>
<summary>网络搜索调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' '{EAS_SERVICE_URL}/api/v1/query/search' -H 'Authorization: EAS_TOKEN' -d '{ "messages": [
        {"role": "user","content": "你好"},
        {"role": "assistant","content": "你好，有什么能帮到您？"},
        {"role": "user", "content": "浙江省会是哪里"}
    ],
    "stream": false}'
  ```

- 返回示例：
  ```json
  {
    "answer": "浙江省的省会是杭州市。如果您还有其他问题或需要更多信息，请告诉我！",
    "session_id": "12eb18e665b244b0b1a6a2245cf71281",
    "docs": null,
    "new_query": "现在浙江省会是哪里"
  }
  ```

</details>

<details>
<summary>数据分析调用示例</summary>

- curl 请求示例：

  ```bash
  curl -X 'POST' '{EAS_SERVICE_URL}/api/v1/query/data_analysis' \
  -H 'Authorization: EAS_TOKEN' \
  -d '{ "messages": [
        {"role": "user","content": "有多少只猫"},
        {"role": "assistant","content": "你有2只猫"},
        {"role": "user", "content": "狗呢"}
    ],
    "stream": true,
    "return_reference": true}'
  ```

- 返回示例：
  ```json
  {
    "delta": "有",
    "is_finished": false
  }
  ```
  ```json
  {
    "delta": "2",
    "is_finished": false
  }
  ```
  ```json
  {
    "delta": "条",
    "is_finished": false
  }
  ```
  ```json
  {
    "delta": "狗",
    "is_finished": false
  }
  ```
  ```json
  {
    "delta": "",
    "is_finished": true,
    "session_id": "b2683d982bf94655ad41970636a7c2ba",
    "new_query": "有多少条狗?",
    "docs": [
      {
        "text": "[(2,)]",
        "score": 1.0,
        "metadata": {
          "query_code_instruction": "SELECT count(*) FROM pets WHERE PetType = 'dog'",
          "query_output": "[(2,)]",
          "col_keys": ["count(*)"],
          "invalid_flag": 0,
          "query_tables": ["pets"]
        },
        "image_url": null
      }
    ]
  }
  ```

</details>
