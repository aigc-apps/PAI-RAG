#!/usr/bin/env python3
"""Smoke test deployed MiniAgent HTTP APIs.

This script covers the three public API families:

1. /v1/chat/completions
2. /v1/responses
3. /v1/runs + /v1/runs/{run_id}/events

Example:
    python scripts/api_smoke.py --base-url http://127.0.0.1:8000

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


DEFAULT_MODEL = "pairag-agent"
DEFAULT_QUERY = "你好，请用一句话介绍你自己，并说明你可以通过 API 被调用。"
DEFAULT_RUN_QUERY = "请确认 /v1/runs 接口可以正常执行，并用一句话说明 Runs API 的作用。"


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


def chat_text(payload):
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


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
        "model": args.model,
        "messages": [{"role": "user", "content": args.query}],
        "stream": False,
    }
    status, resp_headers, data = request_json(
        args.base_url,
        "POST",
        "/v1/chat/completions",
        payload=payload,
        headers=headers,
        timeout=args.timeout,
    )
    require_ok("chat completions", status, data)
    print(f"status={status} session={resp_headers.get('X-Session-Id', '')} run={resp_headers.get('X-Run-Id', '')}")
    print("sample_query:", args.query)
    print("answer:", chat_text(data) or compact_json(data))


def run_responses(args, headers):
    print("\n== /v1/responses ==")
    payload = {
        "model": args.model,
        "input": args.query,
        "stream": False,
    }
    status, resp_headers, data = request_json(
        args.base_url,
        "POST",
        "/v1/responses",
        payload=payload,
        headers=headers,
        timeout=args.timeout,
    )
    require_ok("responses", status, data)
    print(f"status={status} response_id={data.get('id', '')} session={resp_headers.get('X-Session-Id', '')} run={resp_headers.get('X-Run-Id', '')}")
    print("sample_query:", args.query)
    print("answer:", response_text(data) or compact_json(data))
    print("output_types:", [item.get("type") for item in data.get("output") or []])

    if args.multi_turn and data.get("id"):
        follow_up = "继续上一轮，用一句话说明 previous_response_id 如何用于多轮对话。"
        follow_payload = {
            "model": args.model,
            "previous_response_id": data["id"],
            "input": follow_up,
            "stream": False,
        }
        status, _, follow_data = request_json(
            args.base_url,
            "POST",
            "/v1/responses",
            payload=follow_payload,
            headers=headers,
            timeout=args.timeout,
        )
        require_ok("responses follow-up", status, follow_data)
        print("follow_up_query:", follow_up)
        print("follow_up_answer:", response_text(follow_data) or compact_json(follow_data))


def run_runs(args, headers):
    print("\n== /v1/runs + /v1/runs/{run_id}/events ==")
    create_payload = {"input": args.run_query}
    status, resp_headers, run = request_json(
        args.base_url,
        "POST",
        "/v1/runs",
        payload=create_payload,
        headers=headers,
        timeout=args.timeout,
    )
    require_ok("create run", status, run)
    run_id = run.get("run_id")
    session_id = run.get("session_id")
    if not run_id:
        raise RuntimeError(f"create run did not return run_id: {compact_json(run)}")
    print(f"status={status} session={session_id} run={run_id}")
    print("sample_query:", args.run_query)

    output_parts = []
    tool_events = []
    event_count = 0
    started = time.time()
    for event in iter_sse(
        args.base_url,
        "GET",
        f"/v1/runs/{run_id}/events",
        headers=headers,
        timeout=args.timeout,
    ):
        event_count += 1
        data = parse_sse_data(event)
        if isinstance(data, str):
            continue
        event_type = data.get("event")
        if event_type == "message.delta":
            output_parts.append(data.get("delta") or "")
        elif event_type and event_type.startswith("tool."):
            tool_events.append(event_type)
        elif event_type == "ask_user":
            print("ask_user:", compact_json(data))
        elif event_type == "run.completed":
            if data.get("output"):
                output_parts = [data.get("output")]
            print(f"events={event_count} elapsed={time.time() - started:.1f}s tools={tool_events}")
            print("answer:", "".join(output_parts) or compact_json(data))
            break
        elif event_type == "run.failed":
            raise RuntimeError(f"run failed: {compact_json(data)}")
    else:
        raise RuntimeError("run event stream ended before run.completed")

    status, _, status_payload = request_json(
        args.base_url,
        "GET",
        f"/v1/runs/{run_id}",
        headers=headers,
        timeout=args.timeout,
    )
    require_ok("get run", status, status_payload)
    print("run_status:", status_payload.get("status", ""))


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
    parser.add_argument("--base-url", required=True, help="Service base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name, default: {DEFAULT_MODEL}")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Sample query for Chat Completions and Responses")
    parser.add_argument("--run-query", default=DEFAULT_RUN_QUERY, help="Sample query for Runs API")
    parser.add_argument("--auth", default="", help="Authorization header value, for example 'Bearer xxx'")
    parser.add_argument("--header", action="append", help="Extra header, format: 'Name: value'. Can be repeated.")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    parser.add_argument("--multi-turn", action="store_true", help="Also test /v1/responses previous_response_id follow-up")
    parser.add_argument(
        "--only",
        choices=("all", "chat", "responses", "runs"),
        default="all",
        help="Run only one API family",
    )
    args = parser.parse_args(argv)

    headers = build_headers(args)
    print("base_url:", args.base_url.rstrip("/"))
    print("model:", args.model)

    if args.only in ("all", "chat"):
        run_chat_completions(args, headers)
    if args.only in ("all", "responses"):
        run_responses(args, headers)
    if args.only in ("all", "runs"):
        run_runs(args, headers)

    print("\nOK: API smoke test completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
