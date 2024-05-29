# PAI-RAG: An easy-to-use framework for modular RAG.


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
# 启动，支持自定义host(默认0.0.0.0), port(默认8000), config(默认config/demo.yaml)
pai_rag run [--host HOST] [--port PORT] [--config CONFIG_FILE]
```

现在你可以使用命令行向服务侧发送API请求，或者直接打开http://localhost:8000

1.

- **Rag Query请求**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"PAI是什么？"}'
```

- **多轮对话请求**
```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"一键助眠是什么？"}'

# 传入session_id：对话历史会话唯一标识，传入session_id后，将对话历史进行记录，调用大模型将自动携带存储的对话历史。
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"它有什么好处？", "session_id": "5801d0d9-e030-409c-9072-c810b858f9fa"}'

# 传入chat_history：用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间顺序排列。
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"儿童可以使用吗？", "chat_history": [{"user":"一键助眠是什么？", "bot":"一键助眠是一种利用体感振动音乐疗法的睡眠促进技术"}]}'

# 同时传入session_id和chat_history：会用chat_history对存储的session_id所对应的对话历史进行追加更新
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"儿童可以使用吗？", "chat_history": [{"user":"一键助眠是什么？", "bot":"一键助眠是一种利用体感振动音乐疗法的睡眠促进技术"}], "session_id": "5801d0d9-e030-409c-9072-c810b858f9fa"}'
```

- **Agent简单对话**
```bash
curl -X 'POST' http://127.0.0.1:8000/service/query/agent -H "Content-Type: application/json" -d '{"question":"最近互联网公司有发生什么大新闻吗？"}'
```


2. Retrieval Batch评估

```bash
curl -X 'POST' http://127.0.0.1:8000/service/batch_evaluate/retrieval
```

初次调用时会在 localdata/data/evaluation 下面生成一个Retrieval评估数据集（qc_dataset_easy_rag_demo_0.1.1.json， 其中包含了question:context pairs）

返回示例：

```json
{
  "status": 200,
  "eval_resultes": {
    "hit_rate": { "0": 0.821917808219178 },
    "mrr": { "0": 0.6506849315068494 }
  }
}
```

3. Response Batch评估

```bash
curl -X 'POST' http://127.0.0.1:8000/service/batch_evaluate/response
```

初次调用时会在 localdata/data/evaluation 下面生成一个Response评估数据集（qa_dataset_easy_rag_demo_0.1.1.json，其中包含了question:reference_answer pairs）

返回示例：

```json
{
  "status": 200,
  "eval_resultes": {
    "Faithfulness": 0.5,
    "Answer Relevancy": 0.0,
    "Guideline Adherence: The response should fully answer the query.:": 0.5,
    "Guideline Adherence: The response should avoid being vague or ambiguous.:": 0.5,
    "Guideline Adherence: The response should be specific and use statistics or numbers when possible.:": 0.3,
    "Correctness": 0.3,
    "Semantic Similarity": 0.2
  }
}
```

Note: Response Evaluation涉及大量的LLM调用，因此评估过程会耗时较久。

对每一个query，生成answer平均耗时10s左右，评估7个指标平均耗时20s左右。

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
