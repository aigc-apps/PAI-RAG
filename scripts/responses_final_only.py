#!/usr/bin/env python3
"""Call /v1/responses with stream=true and print only the final assistant text.

中间的 SSE 事件（output_text.delta / reasoning_step / output_item.added /
function_call_arguments.delta 等）全部丢弃，只输出最终一条 assistant
message 的文本，便于在脚本里把模型答复当作普通字符串使用。

Usage:
    python scripts/responses_final_only.py
    python scripts/responses_final_only.py --input "你好"
    PAI_RAG_BASE_URL=http://... PAI_RAG_TOKEN=... python scripts/responses_final_only.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = (
    "http://xw-apirag-agent-test-0509-clone.1730760139076263."
    "cn-hangzhou.pai-eas.aliyuncs.com"
)
DEFAULT_TOKEN = "YmYxMDIxNDRmYjMwN2FkYzdjNmQwYTQyZTA4NzVkYjgxODA1MGU4OA=="
DEFAULT_INPUT = (
    "校验引擎配置， 名称： embedding_config, "
    "region/cluster_id: cn-beijing, 环境是生产，status: Released , "
    "instanceId: pairec-cn-inner-khhjd7wnn1geomcirl"
)

TERMINAL_EVENTS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.requires_action",
}


def extract_final_text(response_obj: dict) -> str:
    """Pull the final assistant text out of a `response` object.

    Concatenates every `output_text` chunk under every `message` item.
    """
    parts: list[str] = []
    for item in response_obj.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for chunk in item.get("content", []) or []:
            if chunk.get("type") == "output_text":
                text = chunk.get("text")
                if text:
                    parts.append(text)
    return "".join(parts)


def stream_final_text(base_url: str, token: str, payload: dict, timeout: float) -> int:
    url = base_url.rstrip("/") + "/v1/responses"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": token,
        },
    )

    final_text = ""
    final_status = "unknown"
    final_response_id = ""
    error_obj: dict | None = None
    last_event = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                if not line:
                    last_event = ""
                    continue
                if line.startswith(":"):
                    # SSE comment / keepalive
                    continue
                if line.startswith("event:"):
                    last_event = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    payload_obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = payload_obj.get("type") or last_event
                if event_type not in TERMINAL_EVENTS:
                    # Drop every intermediate event silently.
                    continue

                # Terminal events ship the response object as the top-level
                # data payload (id/status/output live directly on payload_obj).
                # Some servers nest it under "response" — handle both.
                response_obj = payload_obj.get("response") or payload_obj
                final_response_id = response_obj.get("id") or final_response_id
                final_status = response_obj.get("status") or event_type
                text = extract_final_text(response_obj)
                if text:
                    final_text = text
                if event_type == "response.failed":
                    error_obj = response_obj.get("error") or payload_obj.get("error")
    except urllib.error.HTTPError as exc:
        sys.stderr.write(
            f"HTTP {exc.code} {exc.reason}\n{exc.read().decode('utf-8', 'replace')}\n"
        )
        return 2
    except urllib.error.URLError as exc:
        sys.stderr.write(f"network error: {exc}\n")
        return 2

    if error_obj:
        sys.stderr.write(f"response.failed: {json.dumps(error_obj, ensure_ascii=False)}\n")
        return 1
    if not final_text:
        sys.stderr.write(
            f"no final text; status={final_status} response_id={final_response_id}\n"
        )
        return 1

    print(final_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PAI_RAG_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PAI_RAG_TOKEN", DEFAULT_TOKEN),
        help="EAS AccessToken (passed verbatim, no Bearer prefix)",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="user input string")
    parser.add_argument("--model", default=None, help="optional model override")
    parser.add_argument("--conversation", default=None, help="optional conversation id")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    payload: dict = {"input": args.input, "stream": True}
    if args.model:
        payload["model"] = args.model
    if args.conversation:
        payload["conversation"] = args.conversation

    return stream_final_text(args.base_url, args.token, payload, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
