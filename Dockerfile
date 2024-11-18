FROM python:3.11 AS builder

RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY . .

RUN poetry install && rm -rf $POETRY_CACHE_DIR

ENV PYTHON_AGENT_PATH="https://python-agent.oss-rg-china-mainland.aliyuncs.com/1.1.0.rc/aliyun-python-agent.tar.gz"
RUN poetry run aliyun-bootstrap -a install

FROM python:3.11-slim AS prod

RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
WORKDIR /app
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY . .
CMD ["ENABLE_FASTAPI=false", "ENABLE_REQUESTS=false", "ENABLE_AIOHTTPCLIENT=false", "aliyun-instrument", "python" , "src/pai_rag/main.py"]
