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

- 💡 [What is PAI-RAG?](#💡-what-is-pai-rag)
- 🌟 [Key Features](#🌟-key-features)
- 🔎 [Get Started](#🔎-get-started)
  - [Docker](#run-in-docker)
  - [Local](#run-in-local-environment)
- 📜 [Documents](#📜-documents)
  - [API specification](#api-specification)
  - [Agentic RAG](#agentic-rag)
  - [Data Analysis](#data-analysis)

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

## Run in Docker

1. Setup environmental variables.
   ```bash
   cd docker
   cp .env.example .env
   ```
   edit `.env` file if you are using dashscope api or oss store:
2. Start with docker compose command:
   ```bash
   docker compose up -d
   ```
3. Now you can open http://localhost:8000 to check whether it works.

## Run in Local Environment

If you want to start running/developing pai_rag locally, please refer to [local development](./docs/develop/local_develop.md)

# 📜 Documents

## API specification

You can access and integrate our RAG service according to our [API specification](./docs/api.md).

## Agentic RAG

You can use agent with function calling api-tools in PAI-RAG, please refer to the documentation:
[Agentic RAG](./docs/agentic_rag.md)

## Data Analysis

You can use data analysis based on database or sheet file in PAI-RAG, please refer to the documentation: [Data Analysis](./docs/data_analysis_doc.md)

## Parameter Configuration

For more customization options, please refer to the documentation:

[Parameter Configuration Instruction](./docs/config_guide_en.md)

## Supported File Types

| 文件类型     | 文件格式                               |
| ------------ | -------------------------------------- |
| Unstructured | .txt, .docx， .pdf， .html，.pptx，.md |
| Images       | .gif， .jpg，.png，.jpeg， .webp       |
| Structured   | .csv，.xls， .xlsx，.jsonl             |
| Others       | .epub，.mbox，.ipynb                   |

1. .doc files need to be converted to .docx files.
2. .ppt and .pptm files need to be converted to .pptx files.
