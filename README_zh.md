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

# 🌟 主要模块和功能

- 模块化设计，灵活可配置
- 功能丰富，包括Agentic RAG, 多模态问答和nl2sql等
- 基于社区开源组件构建，定制化门槛低
- 多维度自动评估体系，轻松掌握各模块性能质量
- 集成全链路可观测和评估可视化工具
- 交互式UI/API调用，便捷的迭代调优体验
- 阿里云快速场景化部署/镜像自定义部署/开源私有化部署

# 🔎 快速开始

## Docker镜像启动

为了更方便使用，节省较长时间的环境安装问题，我们也提供了直接基于镜像启动的方式。

1. 配置环境变量

   ```bash
   cd docker
   cp .env.example .env
   ```

   如果你需要使用dashscope api或者OSS存储，可以根据需要修改.env中的环境变量。

2. 启动

   ```bash
   docker-compose up -d
   ```

3. 打开浏览器中的 http://localhost:8000 访问web ui.

## 本地启动

如果想在本地启动或者进行代码开发，可以参考文档：[本地运行](./docs/develop/local_develop_zh.md)

# 📜 文档

## API服务

可以直接通过API服务调用RAG能力（上传数据，RAG查询，检索，NL2SQL, Function call等等）。更多细节可以查看[API文档](./docs/api_zh.md)

## Agentic RAG

您也可以在PAI-RAG中使用支持API function calling功能的Agent，请参考文档：
[Agentic RAG](./docs/agentic_rag.md)

## 数据分析 NL2sql

您可以在PAI-RAG中使用支持数据库和表格文件的数据分析功能，请参考文档：[数据分析 Nl2sql](./docs/data_analysis_doc.md)

## 支持文件类型

| 文件类型 | 文件格式                               |
| -------- | -------------------------------------- |
| 非结构化 | .txt, .docx， .pdf， .html，.pptx，.md |
| 图片     | .gif， .jpg，.png，.jpeg， .webp       |
| 结构化   | .csv，.xls， .xlsx，.jsonl             |
| 其他     | .epub，.mbox，.ipynb                   |

1. .doc格式文档需转化为.docx格式
2. .ppt和.pptm格式需转化为.pptx格式
