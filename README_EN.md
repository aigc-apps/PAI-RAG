# A Solution Combining PAI-EAS, AnalyticDB, and LangChain

- Upload local knowledge file and generate embeddings using `SGPT-125M` model.
- Generate and store embeddings in `AnalyticDB` for vector retrieval.
- User posts query, prompt is generated and sent to LLM model service deployed on `PAI-EAS` for real-time question answering.

## Step 1: Development Environment.

### Case 1：Local Environment using Conda.

```bash
conda create --name llm_py310 python=3.10
conda activate llm_py310

git clone git@gitlab.alibaba-inc.com:pai_biz_arch/LLM_Solution.git
cd LLM_Solution

sh install.sh
```

### Case 2：Docker Environment
Pulling an existing Docker environment to prevent unavailability caused by failed environment installation.

```bash
docker pull registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:1.0

sudo docker run -t -d --network host --name llm_docker registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:1.0
docker exec -it llm_docker bash
cd /home/LLM_Solution
```

## Step 2: Edit config.json

- embedding: Path of embedding model, can be customized by user, default is `/embedding_model/SGPT-125M-weightedmean-nli-bitfit`.
- EASCfg: Configuration of LLM model service deployed on `PAI-EAS`, can be customized by user.
- ADBCfg: Environment configuration related to `AnalyticDB`.
- create_docs: Path of knowledge file and related file configuration, default is `/docs`.
- query_topk: Number of relevant results returned by the ADB vector retrieval.
- prompt_template: Prompt customized by user.

## Step 3: Run main.py
1. Upload Local Knowledge File
```bash
python main.py --config config.json --upload True
```

2. Post User Query
```bash
python main.py --config config.json --query "User's questions..."
```

## Demo
```bash
python main.py --config myconfig.json --query 什么是机器学习PAI?

Output:
The answer is:  很抱歉，根据已知信息无法回答该问题。
```

```bash
python main.py --config myconfig.json --upload True 

Output:
Insert into AnalyticDB Success.
```

```bash
python main.py --config myconfig.json --query 什么是机器学习PAI?

Output:
The answer is:  机器学习PAI是阿里云人工智能平台，提供一站式的机器学习解决方案，包括有监督学习、无监督学习和增强学习等。它可以为用户提供从输入特征向量到目标值的映射，帮助用户解决各种机器学习问题，例如商品推荐、用户群体画像和广告精准投放等。
```