import json
from typing import Any, Dict, List, Union
from llama_index.core.utils import resolve_binary
from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)


def to_openai_message_dict(
    message_dict: dict,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
                contents.append(
                    ChatCompletionContentPartTextParam(type=t, text=elem.get("text"))
                )
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
                    ChatCompletionContentPartImageParam(
                        type=t,
                        image_url={
                            "url": image_url,
                            "detail": detail or "auto",
                        },
                    )
                )
            elif t == "tool-call":
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=elem["toolCallId"],
                        type="function",
                        function={
                            "name": elem["toolName"],
                            "arguments": json.dumps(elem["args"], ensure_ascii=False),
                        },
                    )
                )
            elif t == "tool-result":
                call_id = elem["toolCallId"]
                if call_id is None:
                    raise ValueError(
                        "tool_call_id or call_id is required in additional_kwargs for tool messages"
                    )
                message_dict = ChatCompletionToolMessageParam(
                    role=role,
                    content=str(elem["result"]),
                    tool_call_id=call_id,
                )
                return message_dict
    elif isinstance(content, str) or isinstance(content, Dict):
        if role == "system":
            message_dict = ChatCompletionSystemMessageParam(role=role, content=content)
        else:
            message_dict = ChatCompletionAssistantMessageParam(
                role=role, content=json.dumps(content, ensure_ascii=False)
            )

        return message_dict

    if tool_calls:
        message_dict = ChatCompletionAssistantMessageParam(
            role=role,
            content=json.dumps(content, ensure_ascii=False),
            tool_calls=tool_calls,
        )
    else:
        message_dict = ChatCompletionAssistantMessageParam(
            role=role, content=json.dumps(content, ensure_ascii=False)
        )

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
