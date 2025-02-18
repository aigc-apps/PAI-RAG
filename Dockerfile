FROM python:3.11 AS builder

RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY . .

RUN poetry install && rm -rf $POETRY_CACHE_DIR
RUN poetry run aliyun-bootstrap -a install

FROM python:3.11-slim AS prod

RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    ENABLE_FASTAPI=false \
    ENABLE_REQUESTS=false \
    ENABLE_AIOHTTPCLIENT=false \
    ENABLE_HTTPX=false \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && apt-get install -y wget libgl1 libglib2.0-0 libgomp1 curl libgdiplus


RUN wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1w/openssl-1.1.1w.tar.gz && \
    tar -xzvf openssl-1.1.1w.tar.gz && \
    cd openssl-1.1.1w && \
    ./config && \
    make && \
    make install


RUN rm -rf openssl-1.1.1w.tar.gz openssl-1.1.1w

RUN ldconfig

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

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY . .
CMD ["pai_rag", "serve"]
