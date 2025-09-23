FROM python:3.11 AS builder

RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY backend /app/backend
COPY poetry.lock pyproject.toml .
RUN poetry install && rm -rf $POETRY_CACHE_DIR

FROM python:3.11-slim AS prod

WORKDIR /app
COPY model_repository /app/model_repository

RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libgomp1 curl libgdiplus wget perl build-essential nano nodejs npm nginx procps redis-server gettext-base

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# setup paddleocr dependencies
RUN mkdir -p /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer \
 && curl https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar -o /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.tar \
 && tar xvf /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.tar -C /root/.paddleocr/whl/det/ch/

RUN mkdir -p /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer \
 && curl https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar -o /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar \
 && tar xvf /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar -C /root/.paddleocr/whl/rec/ch/

RUN mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer \
 && curl https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -o /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar \
 && tar xvf /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar -C /root/.paddleocr/whl/cls/


COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY resources /app/resources
COPY data /app/data
COPY scripts /app/scripts
COPY frontend /app/frontend
COPY backend /app/backend
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

RUN cd frontend && npm install && npm run build

CMD ["./scripts/start.sh"]
