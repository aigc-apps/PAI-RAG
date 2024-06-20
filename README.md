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
# 启动，支持自定义host(默认0.0.0.0), port(默认8000), config(默认config/demo.yaml)
pai_rag run [--host HOST] [--port PORT] [--config CONFIG_FILE]
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

初次调用时会在 localdata/evaluation 下自动生成一个评估数据集（qc_dataset.json， 其中包含了由LLM生成的query、reference_contexts、reference_node_id、reference_answer）。同时评估过程中涉及大量的LLM调用，因此会耗时较久。也可以通过API接口手动生成QA数据集。

- **（1）全链路效果评估（All）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/evaluate
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
curl -X 'POST' http://127.0.0.1:8000/service/evaluate/retrieval
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
curl -X 'POST' http://127.0.0.1:8000/service/evaluate/response
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

- **（4）手动生成评估的QA数据集（Generate）**

```bash
curl -X 'POST' http://127.0.0.1:8000/service/evaluate/generate
```

返回示例：

```json
{
  "status": 200,
  "result": {
    {
      "examples":[
        {"query":"如何通过\"三原则两注意\"中的方法帮助孩子建立更好的日常生活习惯？","query_by":{"model_name":"qwen-turbo","type":"ai"},"reference_contexts":["\n\n三原则两注意\n\r\n1.想结果:处理问题前，先思考自己想要达成的结果\r\n\r\n2.找重点:不对孩子做过多过细的要求，只要求重点的事情\r\n\r\n3.爱自己:留出时间和空间关爱自己，自己有能量，才能更好地关爱孩子\r\n\r\n1.规律生活:规律生活，可以减少孩子的日常混乱，建议从父母做起\r\n\r\n2.整理房间:房间保持简约、结构化，能增加孩子的专注力和条理性\r\n\r\n\r\n\r"],"reference_node_id":["dacdb33e68b8a60b15044524cfdd710d92b42a1d8c87583bfe8473b2793efc09"],"reference_answer":"通过\"三原则两注意\"中的方法，可以帮助孩子建立更好的日常生活习惯如下：\n\n1. **想结果**：在对孩子提出要求或安排活动时，先考虑这个行为或任务对孩子长期目标的影响。明确你希望他们培养的习惯（如规律作息）对他们的益处，这样孩子会更愿意接受并坚持。\n\n2. **找重点**：避免过度细致的指导，让孩子专注于一两个关键的生活习惯，比如定时睡觉、按时吃饭。这样可以减少他们的压力，让他们更容易理解和执行。\n\n3. **爱自己**：作为父母，要以身作则，展示良好的生活习惯。同时，确保自己有足够的休息和自我照顾，这样你的积极态度和充沛精力会对孩子产生积极影响。\n\n4. **规律生活**：父母自身要保持规律的生活作息，例如设定固定的起床和睡觉时间，这样可以为孩子树立榜样。同时，教导孩子理解规律性对于效率和稳定的重要性。\n\n5. **整理房间**：鼓励孩子参与整理自己的物品，保持环境整洁。一个有序的环境有助于提高孩子的专注力和责任感，也能培养他们的条理性。\n\n通过这些方法，孩子不仅能学会良好的生活习惯，还能在过程中感受到父母的支持和引导，从而更容易形成稳定的日常行为","reference_answer_by":{"model_name":"qwen-turbo","type":"ai"}},
        {"query":"\"如何通过情绪管控三步法来改善家庭教育中的亲子沟通？\"","query_by":{"model_name":"qwen-turbo","type":"ai"},"reference_contexts":["\n\n情绪管控三步一法\n\r\n1.[觉]:觉察并接纳自己第一时间的情绪\r\n2.[停]: 控制自己发泄情绪的行为，比如控制自己不要向孩子吼叫、不要对孩子动手等等\r\n3.[静]:通过离开现场等方式，让自己慢慢冷静\r\n\r\n通过CBT(认知行为疗法)缓解长期的负面情绪\r\n1.让我们有不良情绪的往往不是这件事，而是我们对这件事的认知\r\n\r\n2.运用[自我狡辩法]改变对事情的认知，缓解不良情绪\r\n\r\n\r\n\r"],"reference_node_id":["14f8ea5c9087d2c8821ce2afdbe2065b02c793355aaab907d639d23480e3d9bc"],"reference_answer":"情绪管控三步法（觉、停、静）在家庭教育中的亲子沟通中可以有效地帮助改善和提升沟通质量：\n\n1. **觉** (觉察): 当你发现自己或孩子的情绪开始升温，第一步是觉察到这种情绪的存在。这包括意识到自己的情绪反应以及它可能对孩子产生的影响。了解自己的情绪状态是沟通的第一步。\n\n2. **停** (控制): 在情绪即将失控时，暂停一下，不要立即做出反应。这意味着当你想要吼叫或者对孩子发脾气时，先停下来，避免伤害对方。这一步旨在给双方冷静的时间，避免冲动行为。\n\n3. **静** (冷静): 这个阶段可以通过暂时离开现场、深呼吸、短暂的独处或者进行简单的冷静活动来实现。给自己一些时间让情绪平复下来，这样在重新进入对话时会更加理智和冷静。\n\n通过这个三步法，家长可以更好地管理自己的情绪，以更成熟、理解的态度与孩子交流，从而促进更有效的亲子沟通。同时，CBT（认知行为疗法）的理念也能帮助家长识别并挑战负面思维，调整对孩子的行为和情况的看法，进一步改善家庭氛围。","reference_answer_by":{"model_name":"qwen-turbo","type":"ai"}}
      ]
    }
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
