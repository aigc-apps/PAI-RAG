# Deploy

The combined image (`../Dockerfile`) bundles the whole app in one container:
nginx serves the built SPA and reverse-proxies `/v1` to the uvicorn backend
(`app.lean_main:app`) on the loopback.

## Build & run locally

```bash
# from the repo root
docker build -t pai-loop .
docker run --rm -p 8080:80 \
  -e DASHSCOPE_API_KEY=sk-...            # whichever provider keys config.yaml references \
  -v "$(pwd)/backend/data:/app/backend/data" \   # optional: persist sqlite db / config / skills
  pai-loop
# open http://localhost:8080
```

`backend/data/config.yaml` (env-var-name references only, no secret values) is
baked into the image as the default. Bind-mount `/app/backend/data` to supply
your own config and persist the sqlite database and installed skills.

Build args:
- `INSTALL_PARSERS` (default `true`) — set `false` for the lean base without
  the pdf/docx/pptx/xlsx knowledge-base upload parsers.

Runtime env:
- `WEB_CONCURRENCY` (default `1`) — uvicorn workers. sqlite is single-writer;
  point the backend at Postgres before scaling this up.
- `BACKEND_PORT` (default `8000`) — internal port nginx proxies to.

## CI/CD

- `.github/workflows/ci.yml` — lint + test on push/PR (backend via `uv`,
  frontend via `npm`).
- `.github/workflows/docker.yml` — builds and pushes this image to Aliyun ACR
  on pushes to `feature`/`main` and `v*` tags.

The Docker workflow needs these repository secrets:

| Secret | Example | Purpose |
| --- | --- | --- |
| `ACR_REGISTRY` | `registry.cn-hangzhou.aliyuncs.com` | ACR host |
| `ACR_NAMESPACE` | `pai-loop` | namespace before the repo name |
| `ACR_USERNAME` | — | ACR access username |
| `ACR_PASSWORD` | — | ACR access password / token |
