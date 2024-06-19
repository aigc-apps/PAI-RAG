FROM python:3.10-slim AS builder

RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY . .

RUN poetry install && rm -rf $POETRY_CACHE_DIR

FROM python:3.10-slim AS prod

ENV LC_ALL=C.UTF-8
RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime
ENV Need_Login=1 Run_Environment=WEB

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY . .
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENTRYPOINT ["pai_rag", "run"]
