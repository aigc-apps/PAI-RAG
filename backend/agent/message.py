from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Union

ContentPart = dict  # {"type": "text"|"image_url", ...} for multimodal turns


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as the model emitted it


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Union[str, List[ContentPart], None] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_wire(self) -> dict:
        msg: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def from_wire(cls, d: dict) -> "Message":
        # Wire dicts come from our own to_wire() or the LLM client and are
        # assumed well-formed; missing fields degrade to empty strings, not errors.
        raw_tcs = d.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", "") or "",
            )
            for tc in raw_tcs
        ] or None
        return cls(
            role=d.get("role", ""),
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
        )


def _has_preceding_tool_call(result: List["Message"], tool_call_id: str) -> bool:
    for msg in reversed(result):
        if msg.role == "tool":
            continue
        if msg.role == "assistant" and msg.tool_calls:
            return any(tc.id == tool_call_id for tc in msg.tool_calls)
        return False
    return False


def from_thread(raw: List[dict]) -> List["Message"]:
    """Normalize an incoming thread (mixed dict shapes) into Messages.

    - user content arrays -> flattened text, or text+image parts kept for vision
    - assistant 'tool-call' content parts -> assistant(tool_calls) + tool result pairs
    - tool messages without a matching preceding assistant tool_call -> dropped
    """
    result: List[Message] = []
    for d in raw:
        role = d.get("role", "")
        content = d.get("content")

        if role == "tool":
            tcid = d.get("tool_call_id", "")
            if tcid and _has_preceding_tool_call(result, tcid):
                result.append(Message.from_wire(d))
            continue

        if role == "assistant" and d.get("tool_calls"):
            result.append(Message.from_wire(d))
            continue

        if not isinstance(content, list):
            result.append(Message.from_wire(d))
            continue

        if role == "user":
            text_parts, other_parts = [], []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    other_parts.append(part)
            if other_parts:
                new_content = list(other_parts)
                if text_parts:
                    new_content.insert(0, {"type": "text", "text": "\n".join(text_parts)})
                result.append(Message(role="user", content=new_content))
            else:
                result.append(Message(role="user", content="\n".join(text_parts)))
            continue

        if role == "assistant":
            tc_parts, text_parts = [], []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool-call":
                    tc_parts.append(part)
                elif part.get("type") == "text" and (part.get("text") or "").strip():
                    text_parts.append(part["text"])
            for tc in tc_parts:
                args = tc.get("args", {})
                args_str = (
                    json.dumps(args, ensure_ascii=False)
                    if isinstance(args, dict)
                    else str(args or "{}")
                )
                result.append(
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=tc.get("toolCallId", ""),
                                name=tc.get("toolName", ""),
                                arguments=args_str,
                            )
                        ],
                    )
                )
                tool_result = tc.get("result", "")
                result.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.get("toolCallId", ""),
                        content=(
                            tool_result
                            if isinstance(tool_result, str)
                            else json.dumps(tool_result, ensure_ascii=False)
                        ),
                    )
                )
            if text_parts:
                result.append(Message(role="assistant", content="\n".join(text_parts)))
            elif not tc_parts:
                # empty content list with no tool-calls and no text: skip (matches state.py)
                pass
            continue

        result.append(Message.from_wire(d))
    return result


def keep_last_rounds(msgs: List["Message"], n: int) -> List["Message"]:
    """Keep only the last *n* user-turn rounds (and their following messages)."""
    if n <= 0:
        return msgs
    user_idx = [i for i, m in enumerate(msgs) if m.role == "user"]
    if len(user_idx) <= n:
        return msgs
    return msgs[user_idx[-n]:]
