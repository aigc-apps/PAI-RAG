如果需要在本地进行开发运行，请参考以下步骤：

## 本地启动

1. 克隆仓库

   ```bash
   git clone git@github.com:aigc-apps/PAI-RAG.git
   ```

2. 配置开发环境

   本项目使用poetry进行管理，若在本地环境下使用，建议在安装环境之前先创建一个空环境。为了确保环境一致性并避免因Python版本差异造成的问题，我们指定Python版本为3.11。

   ```bash
   conda create -n rag_env python==3.11
   conda activate rag_env
   ```

   如果使用macOS且需要处理PPTX文件，需要下载依赖库处理PPTX文件

   ```bash
   brew install mono-libgdiplus
   ```

   直接使用poetry安装项目依赖包：

   ```bash
    pip install poetry
    poetry install
    poetry run aliyun-bootstrap -a install
   ```

- 常见网络超时问题

  注：在安装过程中，若遇到网络连接超时的情况，可以添加阿里云或清华的镜像源，在 pyproject.toml 文件末尾追加以下几行：

  ```bash
  [[tool.poetry.source]]
  name = "mirrors"
  url = "http://mirrors.aliyun.com/pypi/simple/" # 阿里云
  # url = "https://pypi.tuna.tsinghua.edu.cn/simple/" # 清华
  priority = "default"
  ```

  之后，再依次执行以下命令：

  ```bash
  poetry lock
  poetry install
  ```

3. 下载其他模型到本地

   ```bash
   # 支持 model name (默认 ""), 没有参数时, 默认下载上述所有模型。
   load_model [--model-name MODEL_NAME]
   ```

4. 启动RAG服务

   使用DashScope API，需要在命令行引入环境变量：

   ```bash
   export DASHSCOPE_API_KEY="xxx"
   ```

   请替换xxx为你自己的DASHSCOPE_API_KEY，DASHSCOPE_API_KEY获取地址为 https://dashscope.console.aliyun.com/apiKey

   启动:

   ```bash
   # 启动，支持自定义host(默认0.0.0.0), port(默认8001), config(默认src/pai_rag/config/settings.yaml), skip-download-models(不加为False)
   # 默认启动时下载模型 [bge-large-zh-v1.5, easyocr] , 可设置 skip-download-models 避免启动时下载模型.
   # 可使用命令行 "load_model" 下载模型 including [bge-large-zh-v1.5, easyocr, SGPT-125M-weightedmean-nli-bitfit, bge-large-zh-v1.5, bge-m3, bge-reranker-base, bge-reranker-large, paraphrase-multilingual-MiniLM-L12-v2, qwen_1.8b, text2vec-large-chinese]
   pai_rag serve [--host HOST] [--port PORT] [--config CONFIG_FILE] [--skip-download-models]
   ```

   ```bash
   pai_rag serve
   ```

5. 启动RAG WebUI

   ```bash
   # 启动，支持自定义host(默认0.0.0.0), port(默认8002), config(默认localhost:8001)
   pai_rag ui [--host HOST] [--port PORT] [rag-url RAG_URL]
   ```

   你也可以打开http://localhost:8002/ 来配置RAG服务以及上传本地数据。

6. 【可选】本地工具-上传数据

   向当前索引存储中插入data_path路径下的新文件

   ```bash
   load_data -c src/pai_rag/config/settings.yaml -d data_path -p pattern
   ```

   path examples:

   ```
   a. load_data -d test/example
   b. load_data -d test/example_data/pai_document.pdf
   c. load_data -d test/example_data -p *.pdf

   ```
