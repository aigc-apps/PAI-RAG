import json
from typing import Dict, List
from llama_index.core.utils import resolve_binary
from llama_index.core.llms import ChatMessage


def to_chat_message(
    message_dict: dict,
) -> ChatMessage:
    assert "role" in message_dict
    assert "content" in message_dict
    role = message_dict["role"]
    content = message_dict.get("content")
    contents = []
    tool_calls = []
    if isinstance(content, list):
        for elem in content:
            t = elem.get("type")
            if t == "text":
                contents.append({"type": t, "text": elem.get("text")})
            elif t == "image_url":
                img = elem["image_url"]["url"]
                detail = elem["image_url"]["detail"]
                if img.startswith("data:"):
                    img_bytes = resolve_binary(raw_bytes=img, as_base64=True).read()
                    img_str = img_bytes.decode("utf-8")
                    image_url = f"base64,{img_str}"
                else:
                    image_url = str(img)
                contents.append(
                    {
                        "type": t,
                        "image_url": {
                            "url": image_url,
                            "detail": detail or "auto",
                        },
                    }
                )
            elif t == "tool-call":
                tool_calls.append(
                    {
                        "id": elem["toolCallId"],
                        "type": "function",
                        "function": {
                            "name": elem["toolName"],
                            "arguments": json.dumps(elem["args"], ensure_ascii=False),
                        },
                    }
                )
            elif t == "tool-result":
                call_id = elem["toolCallId"]
                if call_id is None:
                    raise ValueError(
                        "tool_call_id or call_id is required in additional_kwargs for tool messages"
                    )
                chat_message = ChatMessage(
                    role=role,
                    content=str(elem["result"]),
                    tool_call_id=call_id,
                )
                return chat_message
    elif isinstance(content, str) or isinstance(content, Dict):
        if role == "system":
            chat_message = ChatMessage(role=role, content=content)
        else:
            chat_message = ChatMessage(
                role=role, content=json.dumps(content, ensure_ascii=False)
            )

        return message_dict

    if tool_calls:
        chat_message = ChatMessage(
            role=role,
            content="",
            additional_kwargs={"tool_calls": tool_calls},
        )
    else:
        chat_message = ChatMessage(
            role=role, content=json.dumps(content, ensure_ascii=False)
        )

    return chat_message


def convert_to_chat_messages(messages: List[dict]):
    ret_messages = []
    for message in messages:
        message_dict = to_chat_message(message)
        if isinstance(message_dict, list):
            ret_messages.extend(message_dict)
        else:
            ret_messages.append(message_dict)
    return ret_messages
