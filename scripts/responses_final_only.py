#!/usr/bin/env python3
"""Call /v1/responses with stream=true and stream the final assistant text.

中间的 SSE 事件（reasoning_step / output_item.added / function_call_arguments
等）和内部协议块（<summary> / <thinking> 等）全部丢弃，只把最终一条
assistant message 的可见文本逐 token 流式输出到 stdout。

Usage:
    python scripts/responses_final_only.py
    python scripts/responses_final_only.py --input "你好"
    PAI_RAG_BASE_URL=http://... PAI_RAG_TOKEN=... python scripts/responses_final_only.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


# Internal protocol tags emitted inline by the model. <summary> is metadata;
# <thinking> et al. are reasoning content the React frontend renders in a
# separate "Thinking…" panel — for a CLI consumer they're "intermediate
# process" and should be dropped.
_HIDDEN_TAGS: tuple[str, ...] = (
    "summary",
    "forcing_skill_activation",
    "thinking",
    "checking",
    "taking",
    "working",
    "clinical-thinking",
    "clinical_thinking",
    "taking-action",
    "taking_action",
    "skill-context",
    "skill_context",
)
# Worst-case lookahead: longest "<tagname" + 1 next char (space/`>`/`/`).
_LOOKAHEAD = max(len(t) for t in _HIDDEN_TAGS) + 2

# Used only for the terminal-event fallback (when the server didn't emit
# output_text.delta events at all and we have to strip the whole text in
# one shot). Streaming path uses StreamingTagFilter instead.
_INTERNAL_TAG_RE = re.compile(
    r"<(?P<tag>" + "|".join(re.escape(t) for t in _HIDDEN_TAGS) + r")\b[^>]*>"
    r".*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)


class StreamingTagFilter:
    """Stateful filter that drops <summary> / <thinking> / … blocks from a
    stream of text deltas. Tag boundaries can split across deltas, so the
    filter buffers ambiguous prefixes until it has enough lookahead to decide
    whether a `<` is the start of a hidden tag or just literal text.
    """

    __slots__ = ("_buffer", "_in_tag")

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_tag: str | None = None

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buffer += delta
        out: list[str] = []
        while self._buffer:
            if self._in_tag is not None:
                close = f"</{self._in_tag}>"
                cl = close.lower()
                bl = self._buffer.lower()
                idx = bl.find(cl)
                if idx < 0:
                    # Closing tag not yet present. Retain longest tail that
                    # could be the start of `</tag>`.
                    keep = 0
                    max_keep = min(len(self._buffer), len(close) - 1)
                    for k in range(max_keep, 0, -1):
                        if bl.endswith(cl[:k]):
                            keep = k
                            break
                    self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
                    return "".join(out)
                self._buffer = self._buffer[idx + len(close):]
                self._in_tag = None
                continue

            lt = self._buffer.find("<")
            if lt < 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            if lt > 0:
                out.append(self._buffer[:lt])
                self._buffer = self._buffer[lt:]

            # Buffer now starts with `<`. Decide whether it opens a hidden tag.
            matched: str | None = None
            for tag in _HIDDEN_TAGS:
                opener = f"<{tag}"
                if self._buffer.lower().startswith(opener.lower()):
                    nxt = self._buffer[len(opener):len(opener) + 1]
                    if nxt and nxt not in (" ", ">", "\t", "\n", "\r", "/"):
                        # e.g. <thinking-extra> — not our tag.
                        continue
                    matched = tag
                    break
            if matched is not None:
                close_open = self._buffer.find(">")
                if close_open < 0:
                    # Opening tag not yet complete — wait for more data.
                    return "".join(out)
                self._in_tag = matched
                self._buffer = self._buffer[close_open + 1:]
                continue

            # `<` is not a confirmed hidden-tag opener. If the buffer is too
            # short to rule one out, hold and wait for more bytes.
            if len(self._buffer) < _LOOKAHEAD:
                lower = self._buffer.lower()
                if any(f"<{t}".lower().startswith(lower) for t in _HIDDEN_TAGS):
                    return "".join(out)
            out.append("<")
            self._buffer = self._buffer[1:]
        return "".join(out)

    def flush(self) -> str:
        # Unclosed hidden tag at end of stream — drop silently, better than
        # leaking a half-tag.
        if self._in_tag is not None:
            self._buffer = ""
            self._in_tag = None
            return ""
        out = self._buffer
        self._buffer = ""
        # Drop a trailing fragment that's a strict prefix of any opener
        # (e.g. "<sum" never resolved into "<summary>" — also don't print).
        lower = out.lower()
        for tag in _HIDDEN_TAGS:
            opener = f"<{tag}".lower()
            if opener.startswith(lower) and lower != opener:
                return ""
        return out


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
    """Concatenate every `output_text` chunk under every `message` item, then
    strip internal protocol tags. Used as a fallback when the server didn't
    emit any output_text.delta events on the stream.
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
    return _INTERNAL_TAG_RE.sub("", "".join(parts)).strip()


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

    tag_filter = StreamingTagFilter()
    emitted_any = False
    final_status = "unknown"
    final_response_id = ""
    fallback_text = ""
    error_obj: dict | None = None
    last_event = ""

    def emit(text: str) -> None:
        nonlocal emitted_any
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        emitted_any = True

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                if not line:
                    last_event = ""
                    continue
                if line.startswith(":"):
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
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = obj.get("type") or last_event

                if event_type == "response.output_text.delta":
                    delta = obj.get("delta") or ""
                    if delta:
                        emit(tag_filter.feed(delta))
                    continue

                if event_type not in TERMINAL_EVENTS:
                    continue

                # Terminal event: flush trailing buffer + capture metadata.
                response_obj = obj.get("response") or obj
                final_response_id = response_obj.get("id") or final_response_id
                final_status = response_obj.get("status") or event_type
                emit(tag_filter.flush())
                if not emitted_any:
                    # Server skipped output_text.delta — fall back to the
                    # full text on the response object.
                    fallback_text = extract_final_text(response_obj)
                if event_type == "response.failed":
                    error_obj = response_obj.get("error") or obj.get("error")
    except urllib.error.HTTPError as exc:
        sys.stderr.write(
            f"HTTP {exc.code} {exc.reason}\n{exc.read().decode('utf-8', 'replace')}\n"
        )
        return 2
    except urllib.error.URLError as exc:
        sys.stderr.write(f"network error: {exc}\n")
        return 2

    if error_obj:
        if emitted_any:
            sys.stdout.write("\n")
            sys.stdout.flush()
        sys.stderr.write(f"response.failed: {json.dumps(error_obj, ensure_ascii=False)}\n")
        return 1

    if not emitted_any and fallback_text:
        sys.stdout.write(fallback_text)
        emitted_any = True

    if not emitted_any:
        sys.stderr.write(
            f"no final text; status={final_status} response_id={final_response_id}\n"
        )
        return 1

    sys.stdout.write("\n")
    sys.stdout.flush()
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
