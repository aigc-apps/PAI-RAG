<p align="center">
    <h1>PAI-RAG: An easy-to-use framework for modular RAG </h1>
</p>

[![PAI-RAG CI Build](https://github.com/aigc-apps/PAI-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/aigc-apps/PAI-RAG/actions/workflows/ci.yml)

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a> |
</p>

<details open>
<summary></b>📕 Contents</b></summary>

- 💡 [What is PAI-RAG?](#-what-is-pai-rag)
- 🌟 [Key Features](#-key-features)
- 🔎 [Get Started](#-get-started)
  - [Docker](#run-in-docker)
  - [Local](#run-in-local-environment)
- 📜 [Documents](#-documents)
  - [API specification](#api-specification)
  - [Agentic RAG](#agentic-rag)
  - [Data Analysis](#data-analysis)
  - [Supported File Types](#supported-file-types)

</details>

# 💡 What is PAI-RAG?

PAI-RAG is an easy-to-use opensource framework for modular RAG (Retrieval-Augmented Generation). It combines LLM (Large Language Model) to provide truthful question-answering capabilities, supports flexible configuration and custom development of each module of the RAG system. It offers a production-level RAG workflow for businesses of any scale based on Alibaba Cloud's Platform of Artificial Intelligence (PAI).

# 🎬 PAI-RAG with Web Search Demo (local client using Cherry Studio)

https://github.com/user-attachments/assets/6ea25d2b-dbd5-4013-b337-bd00bd00f41a

# 🌟 Key Features

- Modular design, flexible and configurable
- Powerful RAG capability: [multi-modal rag](docs/multimodal_rag.md), [agentic-rag](docs/agentic_rag.md) and [nl2sql](docs/data_analysis_doc.md) support
- Built on community open source components, low customization threshold
- Multi-dimensional automatic evaluation system, easy to grasp the performance quality of each module
- Integrated llm-based-application tracing and evaluation visualization tools
- Interactive UI/API calls, convenient iterative tuning experience
- Alibaba Cloud fast scenario deployment/image custom deployment/open source private deployment

# 🔎 Get Started

You can run PAI-RAG locally using either a Docker environment or directly from the source code.

## Run with Docker

1. Set up the environmental variables.

   ```bash
   git clone git@github.com:aigc-apps/PAI-RAG.git
   cd PAI-RAG/docker
   cp .env.example .env
   ```

   Edit `.env` file if you are using dashscope api or oss store. See [.env.example](./docker/.env.example) for more details.
   Note you can also configure these settings from our console ui, but it's more safe to configure from environmental variables.

2. Start the Docker containers with the following command
   ```bash
   docker compose up -d
   ```
3. Open your web browser and navigate to http://localhost:8680 to verify that the service is running. The service will need to download the model weights, which may take around 20 minutes.

## Run in a Local Environment

If you prefer to run or develop PAI-RAG locally, please refer to [local development guide](./docs/develop/local_develop.md)

## Simple Query Using the Web UI

1. Open http://localhost:8680 in your web browser. Adjust the index and LLM settings to your preferred models

<img src="docs/figures/quick_start/setting.png" width="600px"/>

2. Go to the "Upload" tab and upload the test data: ./example_data/paul_graham/paul_graham_essay.txt.

<img src="docs/figures/quick_start/upload.png" width="600px"/>

3. Once the upload is complete, switch to the "Chat" tab.

<img src="docs/figures/quick_start/query.png" width="600px"/>

## Simple Query Using the RAG API

1. Open http://localhost:8680 in your web browser. Adjust the index and LLM settings to your preferred models

2. Upload data via API:
   Go to the PAI-RAG base directory

   ```shell
   cd PAI-RAG
   ```

   **Request**

   ```shell
   curl -X 'POST' http://localhost:8680/api/v1/knowledgebases/{knowledgebase_name}/files \
      -H 'Content-Type: multipart/form-data' \
      -F 'files=@example_data/paul_graham/paul_graham_essay.txt'
   ```

   **Response**

   ```json
   {
     "message": "Files have been successfully uploaded."
   }
   ```

   **Note**:
   file is uploaded to RAG service and background job will pick up the file and index it to the vector store.

3. Check the status of the upload job:

   **Request**

   ```shell
   curl -X 'GET' http://localhost:8680/api/v1/knowledgebases/{knowledgebase_name}/history \
   ```

   **Response**

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

4. Perform a RAG query (OpenAI-compatible):

   **Request**

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

   **Response**

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

# 📜 Documents

## API specification

You can access and integrate our RAG service according to our [API specification](./docs/api_v0.3.0.md).

## MultiModal RAG

You can use multimodal RAG to process documents with images, please refer to the documentation: [MultiModal RAG](./docs/multimodal_rag.md)

## Agentic RAG

You can use agent with function calling api-tools in PAI-RAG, please refer to the documentation:
[Agentic RAG](./docs/agentic_rag.md)

## Data Analysis

You can use data analysis based on database or sheet file in PAI-RAG, please refer to the documentation: [Data Analysis](./docs/data_analysis_doc_zh.md)

## Supported File Types

| 文件类型     | 文件格式                               |
| ------------ | -------------------------------------- |
| Unstructured | .txt, .docx， .pdf， .html，.pptx，.md |
| Images       | .gif， .jpg，.png，.jpeg， .webp       |
| Structured   | .csv，.xls， .xlsx，.jsonl             |
| Others       | .epub，.mbox，.ipynb                   |

1. .doc files need to be converted to .docx files.
2. .ppt and .pptm files need to be converted to .pptx files.
