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

# 🌟 Key Features

- Modular design, flexible and configurable
- Powerful RAG capability: multi-modal rag, agentic-rag and nl2sql support
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
3. Open your web browser and navigate to http://localhost:8000 to verify that the service is running. The service will need to download the model weights, which may take around 20 minutes.

## Run in a Local Environment

If you prefer to run or develop PAI-RAG locally, please refer to [local development guide](./docs/develop/local_develop.md)

## Simple Query Using the Web UI

1. Open http://localhost:8000 in your web browser. Adjust the index and LLM settings to your preferred models

<img src="docs/figures/quick_start/setting.png" width="600px"/>

2. Go to the "Upload" tab and upload the test data: ./example_data/paul_graham/paul_graham_essay.txt.

<img src="docs/figures/quick_start/upload.png" width="600px"/>

3. Once the upload is complete, switch to the "Chat" tab.

<img src="docs/figures/quick_start/query.png" width="600px"/>

## Simple Query Using the RAG API

1. Open http://localhost:8000 in your web browser. Adjust the index and LLM settings to your preferred models

2. Upload data via API:
   Go to the PAI-RAG base directory

   ```shell
   cd PAI-RAG
   ```

   **Request**

   ```shell
   curl -X 'POST' http://localhost:8000/api/v1/upload_data \
      -H 'Content-Type: multipart/form-data' \
      -F 'files=@example_data/paul_graham/paul_graham_essay.txt'
   ```

   **Response**

   ```json
   {
     "task_id": "1bcea36a1db740d28194df8af40c7226"
   }
   ```

3. Check the status of the upload job:
   **Request**

   ```shell
   curl 'http://localhost:8000/api/v1/get_upload_state?task_id=1bcea36a1db740d28194df8af40c7226'
   ```

   **Response**

   ```json
   {
     "task_id": "1bcea36a1db740d28194df8af40c7226",
     "status": "completed",
     "detail": null
   }
   ```

4. Perform a RAG query:

   **Request**

   ```shell
   curl -X 'POST' http://localhost:8000/api/v1/query \
      -H "Content-Type: application/json" \
      -d '{"question":"What did the author do growing up?"}'
   ```

   **Response**

   ```json
   {
      "answer":"Growing up, the author worked on writing and programming outside of school. Specifically, he wrote short stories, which he now considers to be awful due to their lack of plot and focus on characters with strong feelings. In terms of programming, he first tried writing programs on an IBM 1401 in 9th grade, using an early version of Fortran. The experience was limited because the only form of input for programs was data stored on punched cards, and he didn't have much data to work with. Later, after getting a TRS-80 microcomputer around 1980, he really started programming by creating simple games, a program to predict the flight height of model rockets, and even a word processor that his father used to write at least one book.",
      "session_id":"ba245d630f4d44a295514345a05c24a3",
      "docs":[
         ...
      ]
   }
   ```

# 📜 Documents

## API specification

You can access and integrate our RAG service according to our [API specification](./docs/api.md).

## Agentic RAG

You can use agent with function calling api-tools in PAI-RAG, please refer to the documentation:
[Agentic RAG](./docs/agentic_rag.md)

## Data Analysis

You can use data analysis based on database or sheet file in PAI-RAG, please refer to the documentation: [Data Analysis](./docs/data_analysis_doc.md)

## Supported File Types

| 文件类型     | 文件格式                               |
| ------------ | -------------------------------------- |
| Unstructured | .txt, .docx， .pdf， .html，.pptx，.md |
| Images       | .gif， .jpg，.png，.jpeg， .webp       |
| Structured   | .csv，.xls， .xlsx，.jsonl             |
| Others       | .epub，.mbox，.ipynb                   |

1. .doc files need to be converted to .docx files.
2. .ppt and .pptm files need to be converted to .pptx files.
