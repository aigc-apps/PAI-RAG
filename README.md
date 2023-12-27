# PAI-Chatbot-Langchain: 基于大语言模型和多向量数据库的知识库问答系统白盒化解决方案

- 支持多种向量数据库: Hologres、Elasticsearch、OpenSearch、AnalyticDB、以及本地FAISS向量库
- 支持多种向量化模型(中文、英文、多语言): SGPT-125M, text2vec-large-chinese, text2vec-base-chinese, paraphrase-multilingual, OpenAIEmbeddings
- 支持任意基于PAI-EAS部署的大模型服务: Qwen, chatglm, llama2, baichuan等系列模型，同时支持ChatGPT调用（需提供OpenAI Key）
- 部署参考链接：[PAI+向量检索快速搭建大模型知识库对话](https://help.aliyun.com/zh/pai/use-cases/use-pai-and-vector-search-to-implement-intelligent-dialogue-based-on-the-foundation-model?spm=a2c4g.11186623.0.0.4510e3efQRyPdt)

## PAI-Chatbot-Langchain白盒化解决方案系统架构图
![SystemArchitecture](html/image.png)
- Step1: 文档处理、切片，针对文本进行不同格式和长度的切分
- Step2: 文本向量化，导入到向量数据库
- Step3: 用户Query向量化，并进行向量相似度检索，获取Top-K条相似文本块
- Step4: 将用户query和Top-K条文本块基于上下文构建Prompt
- Step5: 大模型推理回答，必要时可以finetune模型

### 白盒化自建方案与一体化方案对比

| 维度 | 白盒化自建 | 一体化方案 | 
| ------- | ------- | ------- |
| 模型灵活度 | 支持多种中英文开源模型，如llama2, baichuan, ChatGLM，Qwen，mistral等系列模型，也支持通过API方式调用的模型，比如OpenAI，Gemini各种API | 仅支持内嵌大模型 |
| 模型推理加速 | 支持vLLM、 flash-attention等大模型推理加速框架 | 一般不支持 |
| 向量数据库 | 支持多种向量数据库: Hologres、Elasticsearch、OpenSearch、AnalyticDB、以及本地FAISS向量库 | 仅支持内置 | 
| 业务数据Finetune | 支持 | 一般不支持 |
| Embedding模型 | 支持多种中文/英文/多语言向量模型以及不同的向量维度 | 内置为主，有限的官方和开源模型 |
| 超参数调整 | 支持多种超参数调整，如文档召回参数、模型推理参数 | 有的仅支持temperature和topK |
| Prompt模板 | 提供多种Prompt Template：General, Exreact URL, Accurate Content, 支持用户自定义Prompt| 不支持 |
| 知识库文件格式及上传方式 | 支持多种文件格式：txt、pdf、doc、markdown等, 支持多个文件同时上传, 支持整个文件夹上传 | 文件格式支持txt、doc、pdf、html、json, 只能单个文件上传 |
| 文本处理 | 可根据实际文本情况自定义切块方式: 切块大小 chunk size, 重叠大小 overlap size | 基于段落拆分模型，仅支持默认中文分词器，不能调整 |

## Step 1: 开发环境

### 方案一：本地conda安装

```bash
conda create --name llm_py310 python=3.10
conda activate llm_py310

git clone https://github.com/aigc-apps/LLM_Solution.git
cd LLM_Solution

sh install.sh
pip install --upgrade -r requirements.txt
```

### 方案二：Docker启动

1. 拉取已有的docker环境，防止因环境安装失败导致的不可用
```bash
docker pull registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:env
```

2. 启动docker
```bash
sudo docker run -t -d --network host  --name llm_docker registry.cn-beijing.aliyuncs.com/mybigpai/aigc_apps:env
docker exec -it llm_docker bash
cd /code/LLM_Solution
```

3. 最新代码需要挂载本地目录到docker中

## Step 2: 运行启动WebUI

```bash
uvicorn webui:app --host 0.0.0.0 --port 8000
```
看到如下界面即表示启动成功
![webui](html/webui.jpg)