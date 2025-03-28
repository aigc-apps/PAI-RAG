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

   安装filebrowser

   ```bash
   wget wget https://eas-data.oss-cn-shanghai.aliyuncs.com/3rdparty/sdwebui/filebrowser
   mv filebrowser /bin/filebrowser
   chmod u+x /bin/filebrowser
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

3. 启动RAG服务

   使用DashScope API，需要在命令行引入环境变量：

   ```bash
   export DASHSCOPE_API_KEY="xxx"
   ```

   请替换xxx为你自己的DASHSCOPE_API_KEY，DASHSCOPE_API_KEY获取地址为 https://dashscope.console.aliyun.com/apiKey

   启动:

   ```bash
   # 启动，支持自定义hport(默认8680), worker_num(默认1)
   # 默认启动时下载模型 [bge-m3, pdf-extract]
   # 可使用命令行 "load_model" 下载模型 including [bge-m3, pdf-extract, SGPT-125M-weightedmean-nli-bitfit, bge-large-zh-v1.5, bge-reranker-base, bge-reranker-large, paraphrase-multilingual-MiniLM-L12-v2, qwen_1.8b, text2vec-large-chinese]
   ./scripts/start.sh [-w WORKER_NUM] [-p PORT]
   ```

   ```bash
      ./scripts/start.sh
   ```

   你可以打开http://localhost:8680/ 来配置RAG服务以及上传本地数据。
