# ==============================================================================
# Stage 1: Python dependencies builder
# ==============================================================================
FROM python:3.11-slim AS python-builder

RUN pip3 install --no-cache-dir poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files first for better caching
COPY poetry.lock pyproject.toml ./

# Copy backend code and install dependencies
COPY backend ./backend
RUN poetry install --no-interaction --no-ansi && \
    rm -rf $POETRY_CACHE_DIR && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# Stage 2: Frontend builder
# ==============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app

# Copy frontend files
COPY frontend ./frontend

# Install all dependencies (including dev) and build
RUN cd frontend && npm ci && \
    npm run build && \
    npm ci --omit=dev && \
    rm -rf .next/cache /tmp/*

# ==============================================================================
# Stage 3: Production image
# ==============================================================================
FROM python:3.11-slim AS prod

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    libgdiplus \
    wget \
    perl \
    build-essential \
    nodejs \
    npm \
    nginx \
    procps \
    redis-server \
    gettext-base \
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Harbin /etc/localtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app


# Set up PaddleOCR dependencies in a single layer
RUN mkdir -p /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer \
    /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer \
    /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer \
    && curl -L https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar \
        -o /tmp/ch_PP-OCRv4_det_infer.tar \
    && tar xvf /tmp/ch_PP-OCRv4_det_infer.tar -C /root/.paddleocr/whl/det/ch/ \
    && curl -L https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar \
        -o /tmp/ch_PP-OCRv4_rec_infer.tar \
    && tar xvf /tmp/ch_PP-OCRv4_rec_infer.tar -C /root/.paddleocr/whl/rec/ch/ \
    && curl -L https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar \
        -o /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar \
    && tar xvf /tmp/ch_ppocr_mobile_v2.0_cls_infer.tar -C /root/.paddleocr/whl/cls/ \
    && rm -rf /tmp/*.tar


# Copy virtual environment from builder
COPY --from=python-builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy application files
# Copy built frontend (already has node_modules removed and .next built)
COPY --from=frontend-builder /app/frontend /app/frontend

COPY model_repository ./model_repository
COPY resources ./resources
COPY scripts ./scripts
COPY backend ./backend
COPY alembic ./alembic
COPY alembic.ini ./

# Expose ports
EXPOSE 8680

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8680/health || exit 1

CMD ["./scripts/start.sh"]
