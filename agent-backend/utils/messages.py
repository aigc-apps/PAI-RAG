import json
from typing import Any, Dict, List, Union
from utils.utils import resolve_binary


def to_openai_message_dict(
    message_dict: dict,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    role = message_dict["role"]
    content = message_dict.get("content")
    contents = []
    tool_calls = []
    content_txt = ""
    if isinstance(content, list):
        for elem in content:
            t = elem.get("type")
            if t == "text":
                contents.append({"type": "text", "text": elem.get("text")})
                content_txt += elem.get("text")
            elif t == "image_url":
                img = elem["image_url"]["url"]
                detail = elem["image_url"]["detail"]
                if img.startswith("data:"):
                    img_bytes = resolve_binary(raw_bytes=img, as_base64=True).read()
                    img_str = img_bytes.decode("utf-8")
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"base64,{img_str}",
                                "detail": detail or "auto",
                            },
                        }
                    )
                else:
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": str(img),
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
                            "arguments": json.dumps(elem["args"]),
                        },
                    }
                )
            elif t == "tool-result":
                call_id = elem["toolCallId"]
                if call_id is None:
                    raise ValueError(
                        "tool_call_id or call_id is required in additional_kwargs for tool messages"
                    )
                message_dict = {
                    "role": role,
                    "content": elem["result"],
                    "tool_call_id": call_id,
                }
                return message_dict
    elif isinstance(content, str):
        message_dict = {
            "role": role,
            "content": content,
        }
        return message_dict

    if tool_calls:
        message_dict = {
            "role": role,
            "content": contents,
            "tool_calls": tool_calls,
        }
    else:
        message_dict = {
            "role": role,
            "content": contents,
        }

    return message_dict


def convert_to_openai_messages(messages: List[dict]):
    openai_messages = []
    for message in messages:
        message_dict = to_openai_message_dict(message)
        if isinstance(message_dict, list):
            openai_messages.extend(message_dict)
        else:
            openai_messages.append(message_dict)
    return openai_messages
