# PAI-RAG: An easy-to-use framework for modular RAG.

[![PAI-RAG CI](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml/badge.svg)](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml)

## Get Started

### Step1: Clone Repo

```bash
git clone git@github.com:aigc-apps/PAI-RAG.git
```

### Step2: 配置环境

本项目使用poetry进行管理，建议在安装环境之前先创建一个空环境。为了确保环境一致性并避免因Python版本差异造成的问题，我们指定Python版本为3.10。

```bash
conda create -n rag_env python==3.10
conda activate rag_env
```

使用poetry安装项目依赖包

```bash
pip install poetry
poetry install
```

### Step3: 启动程序

使用OpenAI API，需要在命令行引入环境变量 export OPENAI_API_KEY=""
使用DashScope API，需要在命令行引入环境变量 export DASHSCOPE_API_KEY=""

```bash
# 启动，支持自定义host(默认0.0.0.0), port(默认8000), workers(worker number, default 1)，config(默认src/pai_rag/config/settings.toml)
pai_rag [--host HOST] [--port PORT] [--workers 1] [--config CONFIG_FILE]
```

现在你可以使用命令行向服务侧发送API请求，或者直接打开http://localhost:8000

1. 对话

- **Rag Query请求**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"PAI是什么？"}'
```

- **多轮对话请求**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"PAI是什么？"}'

# 传入session_id：对话历史会话唯一标识，传入session_id后，将对话历史进行记录，调用大模型将自动携带存储的对话历史。
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"它有什么优势？", "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'

# 传入chat_history：用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间顺序排列。
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"它有哪些功能？", "chat_history": [{"user":"PAI是什么？", "bot":"PAI是阿里云的人工智能平台，它提供一站式的机器学习解决方案。这个平台支持各种机器学习任务，包括有监督学习、无监督学习和增强学习，适用于营销、金融、社交网络等多个场景。"}]}'

# 同时传入session_id和chat_history：会用chat_history对存储的session_id所对应的对话历史进行追加更新
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"它有什么优势？", "chat_history": [{"user":"PAI是什么？", "bot":"PAI是阿里云的人工智能平台，它提供一站式的机器学习解决方案。这个平台支持各种机器学习任务，包括有监督学习、无监督学习和增强学习，适用于营销、金融、社交网络等多个场景。"}], "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'
```

- **Agent及调用Fucntion Tool的简单对话**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query/agent -H "Content-Type: application/json" -d '{"question":"今年是2024年，10年前是哪一年？"}'
```

2. 评估

支持三种评估模式：全链路评估、检索效果评估、生成效果评估。

初次调用时会在 localdata/evaluation 下自动生成一个评估数据集（qc_dataset.json， 其中包含了由LLM生成的query、reference_contexts、reference_node_id、reference_answer）。同时评估过程中涉及大量的LLM调用，因此会耗时较久。

- **（1）全链路效果评估（All）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/batch_evaluate
```

返回示例：

```json
{
  "status": 200,
  "result": {
    "batch_number": 6,
    "hit_rate_mean": 1.0,
    "mrr_mean": 0.91666667,
    "faithfulness_mean": 0.8333334,
    "correctness_mean": 4.5833333,
    "similarity_mean": 0.88153079
  }
}
```

- **（2）检索效果评估（Retrieval）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/batch_evaluate/retrieval
```

返回示例：

```json
{
  "status": 200,
  "result": {
    "batch_number": 6,
    "hit_rate_mean": 1.0,
    "mrr_mean": 0.91667
  }
}
```

- **（3）生成效果评估（Response）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/batch_evaluate/response
```

返回示例：

```json
{
  "status": 200,
  "result": {
    "batch_number": 6,
    "faithfulness_mean": 0.8333334,
    "correctness_mean": 4.58333333,
    "similarity_mean": 0.88153079
  }
}
```

3. 上传

支持通过API的方式上传本地文件，并支持指定不同的faiss_path，每次发送API请求会返回一个task_id，之后可以通过task_id来查看文件上传状态（processing、completed、failed）。

- **（1）上传（upload_local_data）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/upload_local_data -H 'Content-Type: multipart/form-data' -F 'file=@local_path/PAI.txt' -F 'faiss_path=localdata/storage'

# Return: {"task_id": "2c1e557733764fdb9fefa063538914da"}
```

- **（2）查看上传状态（upload_local_data）**

```bash
curl http://127.0.0.1:8077/service/get_upload_state\?task_id\=2c1e557733764fdb9fefa063538914da

# Return: {"task_id":"2c1e557733764fdb9fefa063538914da","status":"completed"}
```

### 独立脚本文件：不依赖于整体服务的启动，可独立运行

1. 向当前索引存储中插入新文件

```bash
load_data -d directory_path
```

2. 生成QA评估测试集和效果评估

- type(t): 评估类型，可选，['retrieval', 'response', 'all']，默认为'all'
- overwrite(o): 是否重新生成QA文件，适用于有新增文件的情况，可选 ['True', 'False']，默认为'False'
- file_path(f): 评估结果的输出文件位置，可选，默认为'localdata/evaluation/batch_eval_results.xlsx'

```bash
evaluation -t retrieval -o True -f results_output_path
```
