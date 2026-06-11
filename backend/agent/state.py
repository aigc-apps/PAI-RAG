import json
from typing import List
from pydantic import BaseModel, Field
from common.chat.constants import MessageRole, DEFAULT_AGENT_HISTORY_ROUNDS
from loguru import logger


def _find_preceding_assistant_with_tool_call(result: List[dict], tool_call_id: str) -> bool:
    for msg in reversed(result):
        role = msg.get("role", "")
        if role == "tool":
            continue
        if role == MessageRole.ASSISTANT:
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("id") == tool_call_id:
                    return True
        return False
    return False


def get_message_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        text = ""
        for item in content:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text

    return content


def convert_thread_messages(messages: List[dict]) -> List[dict]:
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id and _find_preceding_assistant_with_tool_call(result, tool_call_id):
                result.append(msg)
            continue

        if role == MessageRole.ASSISTANT and msg.get("tool_calls"):
            result.append(msg)
            continue

        if not isinstance(content, list):
            result.append(msg)
            continue

        if role == MessageRole.USER:
            text_parts = []
            other_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    other_parts.append(part)
            if other_parts:
                new_content = other_parts[:]
                if text_parts:
                    new_content.insert(0, {"type": "text", "text": "\n".join(text_parts)})
                result.append({"role": MessageRole.USER, "content": new_content})
            else:
                result.append({"role": MessageRole.USER, "content": "\n".join(text_parts)})
            continue

        if role == MessageRole.ASSISTANT:
            tool_call_parts = []
            text_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool-call":
                    tool_call_parts.append(part)
                elif part.get("type") == "text":
                    text = part.get("text", "")
                    if text.strip():
                        text_parts.append(text)

            if tool_call_parts:
                for tc in tool_call_parts:
                    tool_call_id = tc.get("toolCallId", "")
                    tool_name = tc.get("toolName", "")
                    args = tc.get("args", {})
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args or "{}")

                    result.append({
                        "role": MessageRole.ASSISTANT,
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": args_str,
                            }
                        }]
                    })
                    tool_result = tc.get("result", "")
                    result.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False),
                    })

            if text_parts:
                result.append({
                    "role": MessageRole.ASSISTANT,
                    "content": "\n".join(text_parts),
                })
            elif not tool_call_parts:
                continue
            continue

        result.append(msg)

    return result


class AgentState(BaseModel):
    messages: List[dict]
    step: int = Field(default=1)

    @classmethod
    def from_messages(
        cls,
        messages: List[dict],
        max_rounds: int = DEFAULT_AGENT_HISTORY_ROUNDS,
    ):
        converted = convert_thread_messages(messages)

        filtered_messages = []
        for message in converted:
            role = message.get("role", "")
            if role == MessageRole.USER:
                filtered_messages.append(message)
            elif role == MessageRole.ASSISTANT:
                filtered_messages.append(message)
            elif role == "tool":
                filtered_messages.append(message)

        trimmed = _keep_last_n_rounds(filtered_messages, max_rounds)
        logger.info(
            f"AgentState: {len(messages)} raw -> {len(converted)} converted -> "
            f"{len(filtered_messages)} filtered -> {len(trimmed)} trimmed (max_rounds={max_rounds})"
        )

        return cls(
            messages=trimmed,
            step=1,
        )


def _keep_last_n_rounds(messages: List[dict], n: int) -> List[dict]:
    if n <= 0:
        return messages

    user_indices = [
        i for i, m in enumerate(messages) if m.get("role") == MessageRole.USER
    ]

    if len(user_indices) <= n:
        return messages

    cut_from = user_indices[-n]
    return messages[cut_from:]
