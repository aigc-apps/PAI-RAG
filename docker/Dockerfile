FROM debian:12-slim AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx

ENV PATH="/root/.local/bin:${PATH}"
RUN pipx install poetry
RUN pipx inject poetry poetry-plugin-bundle
WORKDIR /src
COPY . .
RUN poetry bundle venv --python=/use/bin/python3 --only==main /venv

FROM python:3.10-slim

COPY --from=builder /venv /venv
ENTRYPOINT ["/venv/bin/pai_rag run"]
