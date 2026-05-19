#!/usr/bin/env python3
"""Smoke test deployed MiniAgent HTTP APIs.

This script covers the public /v1/responses and /v1/chat/completions APIs.

Example:
    python scripts/api_smoke.py
    python scripts/api_smoke.py --base-url http://127.0.0.1:8683

With a gateway authorization header:
    python scripts/api_smoke.py \
      --base-url https://example.com \
      --auth "Bearer xxx"
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


DEFAULT_QUERY = "你好，请用一句话介绍你自己，并说明你可以通过 API 被调用。"


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def request_json(base_url, method, path, payload=None, headers=None, timeout=300):
    url = base_url.rstrip("/") + path
    body = None
    req_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            return resp.status, dict(resp.headers), data
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw}
        return exc.code, dict(exc.headers), data


def iter_sse(base_url, method, path, payload=None, headers=None, timeout=300):
    url = base_url.rstrip("/") + path
    body = None
    req_headers = {"Accept": "text/event-stream", **(headers or {})}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_name = None
        data_lines = []
        event_id = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    yield {"event": event_name, "id": event_id, "data": data}
                event_name = None
                data_lines = []
                event_id = None
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("id:"):
                event_id = line[len("id:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())


def parse_sse_data(event):
    raw = event.get("data") or ""
    if raw == "[DONE]":
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def response_text(payload):
    parts = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text") or "")
    return "".join(parts)


def require_ok(name, status, payload):
    if 200 <= status < 300:
        return
    raise RuntimeError(f"{name} failed: HTTP {status} {compact_json(payload)}")


def run_chat_completions(args, headers):
    print("\n== /v1/chat/completions ==")
    payload = {
        "messages": [{"role": "user", "content": args.query}],
        "stream": True,
    }
    if args.model:
        payload["model"] = args.model
    parts = []
    final = None
    event_count = 0
    for event in iter_sse(
        args.base_url, "POST", "/v1/chat/completions",
        payload=payload, headers=headers, timeout=args.timeout,
    ):
        event_count += 1
        data = parse_sse_data(event)
        if data == "[DONE]" or not isinstance(data, dict):
            continue
        final = data
        try:
            delta = data["choices"][0].get("delta") or {}
            if delta.get("content"):
                parts.append(delta["content"])
        except (KeyError, IndexError, TypeError):
            pass
    print(f"events={event_count}")
    print("sample_query:", args.query)
    answer = "".join(parts)
    print("answer:", answer or (compact_json(final) if final else "(no content)"))

    status, _resp_headers, payload = request_json(
        args.base_url, "POST", "/v1/chat/completions",
        payload={
            "messages": [{"role": "user", "content": args.query}],
            "stream": False,
            **({"model": args.model} if args.model else {}),
        },
        headers=headers,
        timeout=args.timeout,
    )
    require_ok("/v1/chat/completions non-stream", status, payload)
    print("non_stream_answer:", payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "(no content)")


def _stream_responses(args, headers, payload):
    """Drive /v1/responses SSE; return (response_id, full_text, completed_payload)."""
    parts = []
    response_id = ""
    completed = None
    event_count = 0
    for event in iter_sse(
        args.base_url, "POST", "/v1/responses",
        payload=payload, headers=headers, timeout=args.timeout,
    ):
        event_count += 1
        data = parse_sse_data(event)
        if data == "[DONE]" or not isinstance(data, dict):
            continue
        etype = data.get("type") or ""
        if etype == "response.created":
            response_id = data.get("id") or response_id
        elif etype == "response.output_text.delta":
            if data.get("delta"):
                parts.append(data["delta"])
        elif etype == "response.completed":
            completed = data.get("response") or data
            response_id = (completed.get("id") if isinstance(completed, dict) else None) or response_id
    return response_id, "".join(parts), completed, event_count


def run_responses(args, headers):
    print("\n== /v1/responses ==")
    payload = {"input": args.query, "stream": True}
    if args.conversation:
        payload["conversation"] = args.conversation
    if args.model:
        payload["model"] = args.model
    response_id, answer, completed, event_count = _stream_responses(args, headers, payload)
    print(f"events={event_count} response_id={response_id}")
    print("sample_query:", args.query)
    if not answer and isinstance(completed, dict):
        answer = response_text(completed)
    print("answer:", answer or "(no content)")
    if isinstance(completed, dict):
        print("output_types:", [item.get("type") for item in completed.get("output") or []])

    if args.multi_turn and response_id:
        follow_up = "继续上一轮，用一句话说明 previous_response_id 如何用于多轮对话。"
        follow_payload = {
            "previous_response_id": response_id,
            "input": follow_up,
            "stream": True,
        }
        if args.model:
            follow_payload["model"] = args.model
        _, follow_answer, follow_completed, _ = _stream_responses(args, headers, follow_payload)
        if not follow_answer and isinstance(follow_completed, dict):
            follow_answer = response_text(follow_completed)
        print("follow_up_query:", follow_up)
        print("follow_up_answer:", follow_answer or "(no content)")


def run_deleted_compat_checks(args, headers):
    print("\n== deleted compatibility endpoints ==")
    status, _resp_headers, payload = request_json(
        args.base_url, "POST", "/v1/runs",
        payload={"input": "ping"},
        headers=headers,
        timeout=args.timeout,
    )
    if status != 404:
        raise RuntimeError(f"/v1/runs expected 404, got HTTP {status} {compact_json(payload)}")
    print("/v1/runs: 404")


def build_headers(args):
    headers = {}
    if args.auth:
        headers["Authorization"] = args.auth
    for item in args.header or []:
        if ":" not in item:
            raise ValueError(f"Invalid --header value, expected 'Name: value': {item}")
        key, value = item.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke test deployed MiniAgent APIs.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8683",
        help="Service base URL, default: http://127.0.0.1:8683",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name; omit to let the backend use its configured default (e.g. qwen-plus locally, pairag-agent via gateway)",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Sample query for Responses")
    parser.add_argument("--conversation", default="", help="Optional Responses conversation id for multi-turn state")
    parser.add_argument("--auth", default="", help="Authorization header value, for example 'Bearer xxx'")
    parser.add_argument("--header", action="append", help="Extra header, format: 'Name: value'. Can be repeated.")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    parser.add_argument("--multi-turn", action="store_true", help="Also test /v1/responses previous_response_id follow-up")
    args = parser.parse_args(argv)

    headers = build_headers(args)
    print("base_url:", args.base_url.rstrip("/"))
    print("model:", args.model or "(backend default)")

    run_chat_completions(args, headers)
    run_responses(args, headers)
    run_deleted_compat_checks(args, headers)

    print("\nOK: API smoke test completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
