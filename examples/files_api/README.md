# `/v1/files` API — local demo

This walks through an end-to-end test of the new `/v1/files` resource:
upload a text / PDF / image / video, then either read inline or search
within the file.

The upload pipeline is async — large uploads finish in the background.
You can either poll `GET /v1/files/{id}` or subscribe to
`GET /v1/files/events?ids=...` (SSE) for status transitions.

## Prerequisites

```bash
# 1. Python deps (Poetry env)
poetry install

# 2. .env (copy from .env.example if needed)
cp .env.example .env
# Edit .env to set:
#   FILE_STORE_TYPE=local
#   DB_TYPE=sqlite3
#   VECTOR_DB_TYPE=local     # for Chroma-backed retrieval; files-API only uses SQL
#   DASHSCOPE_API_KEY=...    # only if you want to test the chat-with-attachment flow

# 3. Redis running on localhost:6379
redis-server --daemonize yes

# 4. Apply migrations
poetry run alembic upgrade head
```

## Start the stack

From the repo root:

```bash
./examples/files_api/run.sh
```

That brings up the API on `:8682` and a Celery worker in the background.
Logs tail to the foreground; Ctrl-C tears everything down.

Alternatively run pieces manually:

```bash
# API (foreground)
cd backend && poetry run uvicorn app.main:app --port 8682

# Worker (separate terminal)
cd backend && poetry run celery -A app.worker worker --loglevel=info
```

## Quick recipes

Set a couple of env vars for the recipes:

```bash
export API=http://localhost:8682
export TENANT=demo-tenant
```

### 1. Upload a small text file → read inline

```bash
# Upload
echo "Hello from PAI-RAG. This is a small text note." > /tmp/note.txt

resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/tmp/note.txt" \
  -F "purpose=chat_attachment")
file_id=$(echo "$resp" | python -c "import json,sys;print(json.load(sys.stdin)['data']['id'])")
echo "file_id=$file_id"

# Poll until ready (worker extracts inline)
until [ "$(curl -s "$API/v1/files/$file_id" -H "X-TENANT-ID: $TENANT" | python -c "import json,sys;print(json.load(sys.stdin)['data']['status'])")" = "succeeded" ]; do
  sleep 1
done

# Read full extracted text
curl -s "$API/v1/files/$file_id/text" -H "X-TENANT-ID: $TENANT" | python -m json.tool
```

Expected: small file → `total_length` tiny, `has_more=false`, `content`
matches what you uploaded.

### 2. Upload a PDF → paginate or search

```bash
# Replace with any PDF you have locally
resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/path/to/big.pdf" \
  -F "purpose=chat_attachment")
file_id=$(echo "$resp" | python -c "import json,sys;print(json.load(sys.stdin)['data']['id'])")

# Wait for extraction (may take a few seconds for large PDFs)
# Or stream status via SSE:
#   curl -N "$API/v1/files/events?ids=$file_id" -H "X-TENANT-ID: $TENANT"

curl -s "$API/v1/files/$file_id" -H "X-TENANT-ID: $TENANT" | python -m json.tool

# --- For short PDFs: read slice-by-slice ---
curl -s "$API/v1/files/$file_id/text?offset=0&limit=5000" -H "X-TENANT-ID: $TENANT" \
  | python -m json.tool

# --- For long PDFs (>5KB extracted text): search ---
curl -s "$API/v1/files/$file_id/chunks?query=conclusion&top_k=3" \
  -H "X-TENANT-ID: $TENANT" | python -m json.tool
```

`total_chunks` in the response tells you whether the file was big enough to
chunk. Zero → it's all served inline, use `/text` instead.

### 3. Upload an image → chat agent analyses it

```bash
resp=$(curl -s -X POST "$API/v1/files" \
  -H "X-TENANT-ID: $TENANT" \
  -F "file=@/path/to/photo.jpg" \
  -F "purpose=chat_attachment")
img_id=$(echo "$resp" | python -c "import json,sys;print(json.load(sys.stdin)['data']['id'])")

# Attach to a chat message — the agent wires up `multimodal-parser` automatically.
curl -sN -X POST "$API/v1/chat/completions" \
  -H "X-TENANT-ID: $TENANT" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "qwen-vl-plus",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "What do you see in this image?",
      "attachments": [
        {"id": "$img_id", "contentType": "image/jpeg"}
      ]
    }
  ]
}
EOF
```

Requires a multimodal LLM to be configured (e.g. Qwen-VL / Claude). Images
and videos never hit the text extractor — they're served raw via
`/v1/files/{id}/content` and the agent base64-encodes them on demand.

### 4. Upload a video → chat agent analyses it

Same as image, with `contentType: "video/mp4"` (or whichever). Agent wires
up the same `multimodal-parser` tool. Video models must be configured.

### 5. Delete when done

```bash
curl -s -X DELETE "$API/v1/files/$file_id" -H "X-TENANT-ID: $TENANT"
```

Files with `purpose=chat_attachment` auto-expire after 7 days and get swept
by the Celery beat GC task (`file_resource_gc_sweep`, hourly by default).

## What gets stored where

| Scope | Where |
|---|---|
| Raw bytes | `file_store` (local FS when `FILE_STORE_TYPE=local`) at `files/{tenant_id}/{YYYYMM}/{file_id}{ext}` |
| File metadata | `pai_file` table |
| Extracted text (all formats) | `pai_file_text_content.content` (up to 500KB) |
| Chunks (files ≥5KB) | `pai_file_chunk` |

## Troubleshooting

- **Status stuck in `pending`** — worker isn't running or can't reach Redis.
  `ps aux | grep celery` should show a worker; otherwise start one with
  `celery -A app.worker worker --loglevel=info`.
- **PDF returns empty text** — check `logs/` for the SimplePdfReader output.
  Scanned PDFs need `ENABLE_MINERU=1` to get OCR.
- **`/chunks` returns `total_chunks: 0`** — file wasn't big enough to chunk.
  Use `/text?offset=&limit=` instead; the full text is already in the
  response to that endpoint.
