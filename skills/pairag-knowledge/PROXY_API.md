# PAI-RAG Skill → Proxy API Contract

This documents every HTTP call the `pairag` skill makes, so you can put a thin
**reverse proxy** in front of PAI-RAG that injects the real credentials. The
skill talks to the proxy with **no token**; the proxy adds the real
`Authorization` (and optionally pins the tenant) and forwards upstream.

```
 pairag CLI ──(no token)──▶  your proxy  ──(+Authorization: Bearer <real token>)──▶  PAI-RAG
 PAIRAG_BASE_URL=https://proxy.internal        injects secret              real backend :8682
```

## Skill side (how to point the skill at the proxy)

```bash
export PAIRAG_BASE_URL=https://proxy.internal   # the proxy, not the real backend
unset  PAIRAG_TOKEN                             # skill holds NO secret
export PAIRAG_TENANT_ID=acme                    # optional; or let the proxy pin it
```

The skill sends, on every request:

| Header | When | Value |
|---|---|---|
| `Accept` | always | `application/json` |
| `Content-Type` | POST only | `application/json` |
| `X-TENANT-ID` | if `PAIRAG_TENANT_ID` set | the tenant id |
| `Authorization` | if `PAIRAG_TOKEN` set | `Bearer <token>` — **omit in proxy mode** |

The skill is **read-only**: it only issues the requests below. A proxy should
**allowlist exactly these** and reject everything else.

---

## Endpoints the skill calls

`{kb}` is a knowledge-base id (32 lowercase hex chars). All query params are
URL-encoded; the skill omits any param that is empty/unset.

### 1. List knowledge bases  — used by `kbs` and name→id resolution
```
GET /v1/config/knowledgebases?size=<int>[&query=<str>]
```
- `size`: page size (skill sends `1000`). `query`: optional name filter.
- **Response** `200`:
  ```json
  {"code":200,"message":"...","data":{
     "items":[{"id":"<kb>","name":"docs","description":"..."}],
     "total":1,"pages":1,"page":1,"size":1000}}
  ```

### 2. Semantic search — used by `search`
```
POST /v1/retrieval
Content-Type: application/json

{"query":"<text>","knowledge_id":"<kb>"}
```
- **Response** `200` — note this one is a **flat** body (no `data` wrapper):
  ```json
  {"records":[
    {"content":"...","score":0.87,"title":"...","url":"...",
     "metadata":{"doc_id":"...","file_path":"...","file_name":"..."}}]}
  ```

### 3. List files (catalog) — used by `catalog`
```
GET /v1/config/knowledgebases/{kb}/files?size=<int>[&query=<str>]
```
- `size`: max rows (from `--limit`). `query`: optional name/title substring.
- **Response** `200`:
  ```json
  {"code":200,"message":"...","data":{
     "items":[{"id":"<file_id>","file_name":"billing.md","status":"succeeded",
               "file_source":"https://...","file_metadata":{"title":"...","source_doc_id":"..."}}],
     "total":1,"pages":1,"page":1,"size":30}}
  ```

### 4. Keyword grep — used by `grep`
```
GET /v1/config/knowledgebases/{kb}/keyword?pattern=<str>&context=<int>&limit=<int>
```
- **Response** `200`:
  ```json
  {"code":200,"message":"...","data":{
     "results":[{"doc_id":"...","file_id":"...","line":42,"match":"...",
                 "context":"...","source_url":"...","title":"..."}],
     "scanned_files":7,"scan_capped":false,"limit_reached":false}}
  ```

### 5. Fetch file content — used by `read`
```
GET /v1/config/knowledgebases/{kb}/file-content?doc_id=<id>[&max_chars=<int>][&offset=<int>]
```
- The skill always sends the id in `doc_id` (the backend resolves a `doc_id`,
  and falls back to treating it as a `file_id`). `offset` defaults to `0`.
- **Response** `200`:
  ```json
  {"code":200,"message":"...","data":{
     "file_id":"...","file_name":"...","title":"...","source_url":"...","doc_id":"...",
     "content":"...","content_length":20007,"offset":0,"returned_chars":6000,
     "truncated":true,"next_offset":6000,"degraded":null,"metadata":{}}}
  ```

### Allowlist summary

| # | Method | Path pattern | Required query / body |
|---|---|---|---|
| 1 | GET  | `/v1/config/knowledgebases` | `size`, `query?` |
| 2 | POST | `/v1/retrieval` | body `{query, knowledge_id}` |
| 3 | GET  | `/v1/config/knowledgebases/{kb}/files` | `size`, `query?` |
| 4 | GET  | `/v1/config/knowledgebases/{kb}/keyword` | `pattern`, `context`, `limit` |
| 5 | GET  | `/v1/config/knowledgebases/{kb}/file-content` | `doc_id`, `max_chars?`, `offset?` |

---

## Proxy responsibilities

1. **Inject the secret.** Add `Authorization: Bearer <REAL_TOKEN>` from the
   proxy's own config. **Strip any client-supplied `Authorization`** (never trust it).
2. **(Optional) pin the tenant.** If multi-tenant, set/overwrite `X-TENANT-ID`
   to the tenant the caller is allowed to see, instead of trusting the client's.
3. **Forward** method, path, query string, and (for `POST /v1/retrieval`) the
   JSON body unchanged to the real backend.
4. **Pass the response through verbatim** — same status code and body. The skill
   relies on:
   - the `{code,message,data}` envelope for endpoints 1/3/4/5 and the **flat**
     `{records}` body for endpoint 2;
   - the HTTP **status code** (it shows a friendly error on ≥400 and reads
     `message`/`detail` from the JSON error body).
   Do not rewrite or re-wrap bodies.
5. **Deny by default.** Reject any path/method not in the allowlist (the skill is
   read-only; this keeps the proxy from exposing writes or other APIs).

---

## Reference implementations

### nginx

```nginx
# Real token kept only here, on the proxy host.
map $request_method $allowed_retrieval { POST 1; default 0; }

server {
  listen 443 ssl;
  server_name proxy.internal;

  # Inject the secret; never accept one from the client.
  proxy_set_header Authorization "Bearer REPLACE_WITH_REAL_TOKEN";
  proxy_set_header X-TENANT-ID   "acme";          # optional: pin tenant
  proxy_set_header Host          $host;

  set $upstream https://pairag.internal:8682;

  # 2. POST /v1/retrieval
  location = /v1/retrieval {
    if ($allowed_retrieval = 0) { return 405; }
    proxy_pass $upstream;
  }

  # 1. GET /v1/config/knowledgebases   (exact match, list only)
  location = /v1/config/knowledgebases {
    limit_except GET { deny all; }
    proxy_pass $upstream;
  }

  # 3/4/5. GET /v1/config/knowledgebases/{kb}/(files|keyword|file-content)
  location ~ "^/v1/config/knowledgebases/[0-9a-f]{32}/(files|keyword|file-content)$" {
    limit_except GET { deny all; }
    proxy_pass $upstream;
  }

  # Everything else is denied.
  location / { return 403; }
}
```

### FastAPI (httpx)

```python
import os, httpx
from fastapi import FastAPI, Request, Response, HTTPException

UPSTREAM = os.environ["PAIRAG_UPSTREAM"].rstrip(
    "/"
)  # https://pairag.internal:8682
TOKEN = os.environ["PAIRAG_TOKEN"]  # the real secret, only here
TENANT = os.environ.get("PAIRAG_TENANT_ID")  # optional pin

app = FastAPI()
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=60)

ALLOW = {
    ("GET", "/v1/config/knowledgebases"),
    ("POST", "/v1/retrieval"),
}
import re

KB_SUB = re.compile(
    r"^/v1/config/knowledgebases/[0-9a-f]{32}/(files|keyword|file-content)$"
)


def _allowed(method: str, path: str) -> bool:
    return (method, path) in ALLOW or (
        method == "GET" and bool(KB_SUB.match(path))
    )


@app.api_route("/{full_path:path}", methods=["GET", "POST"])
async def proxy(full_path: str, request: Request):
    path = "/" + full_path
    if not _allowed(request.method, path):
        raise HTTPException(status_code=403, detail="not allowed")
    # Inject our secret; drop anything the client sent.
    headers = {"Accept": "application/json"}
    if request.method == "POST":
        headers["Content-Type"] = "application/json"
    headers["Authorization"] = f"Bearer {TOKEN}"
    if TENANT:
        headers["X-TENANT-ID"] = TENANT
    upstream = await client.request(
        request.method,
        path,
        params=request.query_params,
        content=await request.body(),
        headers=headers,
    )
    # Pass status + body through unchanged.
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
```

Point the skill at it: `PAIRAG_BASE_URL=https://proxy.internal` with no `PAIRAG_TOKEN`.
