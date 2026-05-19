"""PAI-RAG API 客户端调用示例（纯标准库，零依赖）。

推荐使用 /v1/responses；通用 OpenAI Chat Completions 客户端可使用
/v1/chat/completions。/v1/sessions 仅用于当前 Web 前端式的会话创建、
查询、删除；发起 Responses 调用时把 session_id 放进 conversation 字段。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from contextlib import closing
from typing import Any, Callable, Iterator


class APIError(RuntimeError):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        message = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)
        super().__init__(f"{status}: {message}")


def http_request(
    method: str,
    url: str,
    body: dict | None = None,
    *,
    stream: bool = False,
    timeout: int = 300,
):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        raise APIError(exc.code, payload) from None

    headers = {key.lower(): value for key, value in resp.headers.items()}
    if stream:
        return resp.status, headers, resp
    raw = resp.read().decode("utf-8", errors="replace")
    resp.close()
    return resp.status, headers, json.loads(raw) if raw else None


def iter_sse(resp) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_buf: list[str] = []

    def flush():
        nonlocal event_name, data_buf
        if data_buf:
            yield event_name, "\n".join(data_buf)
        event_name = ""
        data_buf = []

    with closing(resp):
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                yield from flush()
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_buf.append(line[len("data:"):].lstrip())
        yield from flush()


def response_text(payload: dict) -> str:
    parts: list[str] = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text") or "")
    return "".join(parts)


def stream_response(base_url: str, body: dict) -> tuple[str, str, dict | None]:
    _, _headers, resp = http_request("POST", f"{base_url}/v1/responses", body=body, stream=True)
    response_id = ""
    parts: list[str] = []
    terminal: dict | None = None
    for event_name, data in iter_sse(resp):
        if data == "[DONE]":
            break
        payload = json.loads(data)
        event_type = payload.get("type") or event_name
        if event_type == "response.created":
            response_id = payload.get("id") or response_id
            print(f"[created] {response_id}")
        elif event_type == "response.output_text.delta":
            delta = payload.get("delta") or ""
            parts.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()
        elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
            terminal = payload.get("response") if isinstance(payload.get("response"), dict) else payload
            response_id = (terminal or {}).get("id") or response_id
    if parts:
        sys.stdout.write("\n")
    return response_id, "".join(parts), terminal


def demo_chat_completions(base_url: str, model: str) -> None:
    print("--- /v1/chat/completions ---")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "stream": False,
    }
    _, _, payload = http_request("POST", f"{base_url}/v1/chat/completions", body=body)
    print(payload["choices"][0]["message"]["content"])


def demo_chat_completions_stream(base_url: str, model: str) -> None:
    print("--- /v1/chat/completions stream ---")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "stream": True,
    }
    _, _headers, resp = http_request("POST", f"{base_url}/v1/chat/completions", body=body, stream=True)
    for _event_name, data in iter_sse(resp):
        if data == "[DONE]":
            break
        payload = json.loads(data)
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
    sys.stdout.write("\n")


def demo_responses_stream(base_url: str, model: str) -> None:
    print("--- /v1/responses stream ---")
    body: dict[str, Any] = {
        "model": model,
        "input": "用一句话介绍你自己，并说明你可以通过 API 被调用。",
        "stream": True,
    }
    response_id, text, terminal = stream_response(base_url, body)
    if not text and terminal:
        text = response_text(terminal)
    print(f"response_id: {response_id}")
    print(f"answer: {text or '(no content)'}")


def demo_responses_conversation(base_url: str, model: str) -> None:
    print("--- /v1/responses conversation ---")
    _, _, session = http_request("POST", f"{base_url}/v1/sessions", body={})
    conversation = session["session_id"]
    print(f"conversation: {conversation}")

    for prompt in ("记住一个数字：7。然后回复“好的”。", "我刚才让你记的数字是什么？"):
        print(f"\nuser: {prompt}")
        body = {
            "model": model,
            "conversation": conversation,
            "input": prompt,
            "stream": True,
        }
        _response_id, text, terminal = stream_response(base_url, body)
        if not text and terminal:
            text = response_text(terminal)
        print(f"assistant: {text or '(no content)'}")


def demo_previous_response_id(base_url: str, model: str) -> None:
    print("--- /v1/responses previous_response_id ---")
    first_id, _text, _terminal = stream_response(base_url, {
        "model": model,
        "input": "回复一个词：alpha",
        "stream": True,
    })
    print(f"first_response_id: {first_id}")
    second_id, text, terminal = stream_response(base_url, {
        "model": model,
        "previous_response_id": first_id,
        "input": "继续上一轮，回复一个词：beta",
        "stream": True,
    })
    if not text and terminal:
        text = response_text(terminal)
    print(f"second_response_id: {second_id}")
    print(f"answer: {text or '(no content)'}")


def demo_sessions(base_url: str, _model: str) -> None:
    print("--- /v1/sessions ---")
    _, _, created = http_request("POST", f"{base_url}/v1/sessions", body={})
    sid = created["session_id"]
    print(f"created: {sid}")
    _, _, detail = http_request("GET", f"{base_url}/v1/sessions/{sid}")
    print(f"status: {detail['status']} pending_hitl={detail.get('pending_hitl')!r}")
    _, _, deleted = http_request("DELETE", f"{base_url}/v1/sessions/{sid}")
    print(f"deleted: {deleted['deleted']}")


def demo_deleted_compat(base_url: str, _model: str) -> None:
    print("--- deleted compatibility endpoints ---")
    for method, path, body in (
        ("POST", "/v1/runs", {"input": "ping"}),
    ):
        try:
            http_request(method, f"{base_url}{path}", body=body)
        except APIError as exc:
            print(f"{path}: HTTP {exc.status}")
            continue
        raise RuntimeError(f"{path} unexpectedly succeeded")


DEMOS: dict[str, Callable[[str, str], None]] = {
    "chat": demo_chat_completions,
    "chat-stream": demo_chat_completions_stream,
    "responses-stream": demo_responses_stream,
    "conversation": demo_responses_conversation,
    "previous-response": demo_previous_response_id,
    "sessions": demo_sessions,
    "deleted-compat": demo_deleted_compat,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PAI-RAG Responses API 调用示例")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--only", choices=list(DEMOS), help="只跑指定的一节")
    args = parser.parse_args()

    names = [args.only] if args.only else list(DEMOS)
    for name in names:
        print()
        print("=" * 70)
        print(f"  {name}")
        print("=" * 70)
        DEMOS[name](args.base_url.rstrip("/"), args.model)


if __name__ == "__main__":
    main()
