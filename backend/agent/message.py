from __future__ import annotations

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
