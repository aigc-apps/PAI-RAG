# 结合PAI-EAS、PAI-DSW、LangChain 结合向量检索库（Hologres / AnalyticDB / Elasticsearch / FAISS）的解决方案

- 上传用户本地知识库文件，基于SGPT-125M模型生成embedding
- 生成embedding存储到向量数据库，并用于后续向量检索
- 输入用户问题，输出该问题的prompt，用于后续PAI-EAS-LLM部分生成答案
- 将产生的prompt送入EAS部署的LLM模型服务，实时获取到问题的答案
- 支持多种阿里云数据库（如Hologres、AnalyticDB、Elasticsearch）及本地FAISS向量库

## Step 1: 开发环境

### 方案一：本地conda安装

```bash
conda create --name llm_py310 python=3.10
conda activate llm_py310

git clone git@gitlab.alibaba-inc.com:pai_biz_arch/LLM_Solution.git
cd LLM_Solution

sh install.sh
pip install --upgrade -r requirements.txt
```

### 方案二：Docker启动

1. 拉取已有的docker环境，防止因环境安装失败导致的不可用
```bash
docker pull registry.cn-beijing.aliyuncs.com/mybigpai/chatbot_langchain:2.3
```

2. 克隆项目
```bash
git clone git@gitlab.alibaba-inc.com:pai_biz_arch/LLM_Solution.git
cd LLM_Solution
```

3. 将本地项目挂载到docker并启动
```bash
sudo docker run -t -d --network host  --name llm_docker -v $(pwd):/root/LLM_Solution registry.cn-beijing.aliyuncs.com/mybigpai/chatbot_langchain:2.3
docker exec -it llm_docker bash
cd /root/LLM_Solution
```

### 方案三：使用PAI-DSW一键拉起

1. 进入PAI-DSW官网：https://pai.console.aliyun.com/notebook，新建一个实例

2. 在镜像处选择“镜像URL”：填入 registry.cn-beijing.aliyuncs.com/mybigpai/chatbot_langchain:2.3

3. 确认后等待环境资源准备完毕后启动即可

4. 进入DSW实例，选择“打开”，在IDE处进入"/code/LLM_Solution"文件夹下即可编辑代码

## Step 2: 配置config.json

- embedding: embedding模型路径，可以用户自定义挂载，默认使用`embedding_model/SGPT-125M-weightedmean-nli-bitfit`。
- EASCfg: 配置已部署在`PAI-EAS`上LLM模型服务，可以用户自定义
- ADBCfg（可选）: AnalyticDB相关环境配置
- HOLOCfg（可选）: Hologres相关环境配置
- ElasticSearchCfg（可选）: ElasticSearch相关环境配置
- 注：如果不配置以上三种，则默认使用`FAISS`存储在本地根目录`/faiss_index`下（适合数据量很少的情况）
- create_docs: 知识库路径和相关文件配置，默认使用`/docs`下的所有文件
- query_topk: 检索返回的相关结果的数量
- prompt_template: 用户自定义的`prompt`

## Step 3: 运行启动WebUI

```bash
uvicorn webui:app --host 0.0.0.0 --port 8000
```
看到如下界面即表示启动成功
![webui](html/webui.jpg)
