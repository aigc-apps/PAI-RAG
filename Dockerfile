FROM python:3.11 AS builder

RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY . .

RUN poetry install
RUN pip3 cache purge
RUN pip3 install https://miropsota.github.io/torch_packages_builder/detectron2/#:~:text=detectron2%2D0.6%2B2a420edpt2.3.0cpu%2Dcp311%2Dcp311%2Dlinux_x86_64.whl
RUN rm -rf $POETRY_CACHE_DIR

FROM python:3.11-slim AS prod

RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY . .
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
CMD ["pai_rag", "run"]
