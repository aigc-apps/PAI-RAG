<p align="center">
    <h1>PAI-RAG: 一个易于使用的模块化RAG框架 </h1>
</p>

[![PAI-RAG CI](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml/badge.svg)](https://github.com/aigc-apps/PAI-RAG/actions/workflows/main.yml)

<details open>
<summary></b>📕 目录</b></summary>

- 💡 [什么是PAI-RAG?](#-什么是pai-rag)
- 🌟 [主要模块和功能](#-主要模块和功能)
- 🔎 [快速开始](#-快速开始)
  - [Docker镜像](#Docker镜像启动)
  - [本地环境](#本地启动)
- 📜 [文档](#-文档)
  - [API服务](#api服务)
  - [Agentic RAG](#agentic-rag)
  - [数据分析Nl2sql](#数据分析-nl2sql)
  - [支持文件类型](#支持文件类型)

</details>

# 💡 什么是PAI-RAG?

PAI-RAG 是一个易于使用的模块化 RAG（检索增强生成）开源框架，结合 LLM（大型语言模型）提供真实问答能力，支持 RAG 系统各模块灵活配置和定制开发，为基于阿里云人工智能平台（PAI）的任何规模的企业提供生产级的 RAG 系统。

# 🎬 PAI-RAG联网搜索的效果展示（本地客户端使用开源的Cherry Studio）

https://github.com/user-attachments/assets/6ea25d2b-dbd5-4013-b337-bd00bd00f41a

# 🌟 主要模块和功能

- 模块化设计，灵活可配置
- 功能丰富，包括[Agentic RAG](docs/agentic_rag.md), [多模态问答](docs/multimodal_rag.md)和[nl2sql](docs/data_analysis_doc.md)等
- 基于社区开源组件构建，定制化门槛低
- 多维度自动评估体系，轻松掌握各模块性能质量
- 集成全链路可观测和评估可视化工具
- 交互式UI/API调用，便捷的迭代调优体验
- 阿里云快速场景化部署/镜像自定义部署/开源私有化部署

# 🔎 快速开始

## Docker镜像启动

您可以通过两种方式在本地运行 PAI-RAG：Docker 环境或直接从源代码运行。

1. 设置环境变量

   ```bash
   git clone git@github.com:aigc-apps/PAI-RAG.git
   cd PAI-RAG/docker
   cp .env.example .env
   ```

   如果您需要使用通义千问API或者阿里云OSS存储，请编辑 .env 文件。
   其中DASHSCOPE_API_KEY获取地址为 https://dashscope.console.aliyun.com/apiKey。
   当服务启动后您依然可以在WEB UI中配置这些API_KEY信息，但是我们建议您通过环境变量的方式配置。

2. 使用`docker compose`命令启动服务：

   ```bash
   docker-compose up -d
   ```

3. 打开浏览器中的 http://localhost:8680 访问web ui. 第一次启动服务会下载需要的相关模型文件，需要等待20分钟左右。

## 本地启动

如果想在本地启动或者进行代码开发，可以参考文档：[本地开发指南](./docs/develop/local_develop_zh.md)

## 通过Web UI查询的示例

1. 打开 http://localhost:8680 在浏览器中。根据需要调整索引和LLM设置。

   <img src="docs/figures/quick_start/setting.png" width="600px"/>

2. 访问"上传"页面，上传测试数据：./example_data/paul_graham/paul_graham_essay.txt。

   <img src="docs/figures/quick_start/upload.png" width="600px"/>

3. 切换到"聊天"页面, 进行对话。

   <img src="docs/figures/quick_start/query.png" width="600px"/>

## 通过API接口查询的示例

1. 打开 http://localhost:8680 在浏览器中。根据需要调整索引和LLM设置。

2. 使用API上传数据：

   切换到`PAI-RAG`目录

   ```shell
   cd PAI-RAG
   ```

   **请求**

   ```shell
   curl -X 'POST' http://localhost:8680/api/v1/knowledgebases/{knowledgebase_name}/files \
      -H 'Content-Type: multipart/form-data' \
      -F 'files=@example_data/paul_graham/paul_graham_essay.txt'
   ```

   **响应**

   ```json
   {
     "message": "Files have been successfully uploaded."
   }
   ```

3. 检查上传任务的状态：

   **请求**

   ```shell
   curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/{knowledgebase_name}/history \
   ```

   **响应**

   ```json
   [
     {
       "task_id": "93d3782ccd4b33afdc1b6a1f0ce18e3a",
       "operation": "ADD",
       "file_name": "localdata/knowledgebase/default/docs/paul_graham_essay.txt",
       "status": "done",
       "last_modified_time": "2025-03-28 15:47:07"
     }
   ]
   ```

4. OpenAI接口兼容的查询:

   **请求**

   ```shell
   curl -X 'POST' http://localhost:8680/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "default",
      "messages": [
         {"role": "user", "content": "杭州在中国哪个省?"}
      ],
      "stream":false,
   }'
   ```

   **响应**

   ```json
   {
     "id": "7aac074feef14c31a322b15bb1c4c452",
     "choices": [
       {
         "finish_reason": "stop",
         "index": 0,
         "logprobs": null,
         "message": {
           "content": "杭州位于中国的**浙江省**。它是浙江省的省会城市，也是中国著名的历史文化名城和旅游胜地，以西湖、龙井茶等闻名于世。",
           "refusal": null,
           "role": "assistant",
           "audio": null,
           "function_call": null,
           "tool_calls": null
         }
       }
     ],
     "created": 1743148661,
     "model": "DeepSeek-V3",
     "object": "chat.completion",
     "service_tier": null,
     "system_fingerprint": null,
     "usage": {
       "completion_tokens": 46,
       "prompt_tokens": 2114,
       "total_tokens": 2160,
       "completion_tokens_details": null,
       "prompt_tokens_details": null
     },
     "citation_details": [],
     "citations": []
   }
   ```

# 📜 文档

## API服务

可以直接通过API服务调用RAG能力（上传数据，RAG查询，检索，NL2SQL, Function call等等）。更多细节可以查看[API文档](./docs/api_v0.3.0.md)

## 多模态问答

可以配置OSS存储和多模态大语言模型来实现文档中的图片理解和问答。请参考问答[多模态问答](./docs/multimodal_rag.md)

## Agentic RAG

您也可以在PAI-RAG中使用支持API function calling功能的Agent，请参考文档：
[Agentic RAG](./docs/agentic_rag.md)

## 数据分析 Nl2sql

您可以在PAI-RAG中使用支持数据库和表格文件的数据分析功能，请参考文档：[数据分析 Nl2sql](./docs/data_analysis_doc_zh.md)

## 支持文件类型

| 文件类型 | 文件格式                               |
| -------- | -------------------------------------- |
| 非结构化 | .txt, .docx， .pdf， .html，.pptx，.md |
| 图片     | .gif， .jpg，.png，.jpeg， .webp       |
| 结构化   | .csv，.xls， .xlsx，.jsonl             |
| 其他     | .epub，.mbox，.ipynb                   |

1. .doc格式文档需转化为.docx格式
2. .ppt和.pptm格式需转化为.pptx格式
