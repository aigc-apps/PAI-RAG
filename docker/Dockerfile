FROM filebrowser/filebrowser as filebrowser
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# 为了避免时区选择交互
ENV DEBIAN_FRONTEND=noninteractive

# 安装基本工具和编译依赖项
RUN apt-get update -y && apt-get install -y  \
    libgl1 libglib2.0-0 \
    sudo wget tar git git-lfs vim tig \
    telnet curl net-tools zip\
    python3-pip python-is-python3 python3.10-venv libpq-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# 复制代码到容器中 (根据需要调整或移除此行) [新加坡无法连接gitlab，因此将代码打包上传到OSS再下载]
# RUN git clone --branch master http://gitlab-ci-token:of-xxx@gitlab.alibaba-inc.com/pai_biz_arch/EasyRAG.git /app/EasyRAG

# 若测试阶段，可以自行通过脚本打包代码上传至OSS，并替换下面的tar包名称
ADD https://pai-rag.oss-cn-hangzhou.aliyuncs.com/codes/EasyRAG_yyyymmdd-hhmmss-N.tar.gz /app/
RUN cd /app/ && mkdir EasyRAG && tar -xvf EasyRAG_yyyymmdd-hhmmss-N.tar.gz -C EasyRAG/

# 设置工作目录
WORKDIR /app/EasyRAG

# 进入工作目录并进行环境安装
RUN cd /app/EasyRAG
RUN pip install poetry \
    && poetry config virtualenvs.create false
RUN poetry install
RUN cd src/pai_rag/integrations/llms/paieas && poetry install
RUN cd /app/EasyRAG/package && pip install pai_llm_trace-0.0.1-py3-none-any.whl

RUN cd /app/EasyRAG
