import json
from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    TextBlock,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


def to_chat_message(
    message_dict: dict,
) -> ChatMessage:
    assert "role" in message_dict
    assert "content" in message_dict
    role = message_dict["role"]
    content = message_dict.get("content")
    tool_calls = []
    role = message_dict.get("role")
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content")
    blocks = []
    tool_call_index = 0
    if isinstance(content, str):
        return ChatMessage(role=role, content=content)
    else:
        # list
        for elem in content:
            t = elem.get("type")
            if t == "text":
                blocks.append(TextBlock(text=elem.get("text")))
            elif t == "image_url":
                img = elem.get("image_url").get("url")
                detail = elem.get("image_url").get("detail", "auto")
                if img.startswith("data:"):
                    blocks.append(ImageBlock(image=img, detail=detail))
                else:
                    blocks.append(ImageBlock(url=img, detail=detail))
            elif t == "tool-call":
                tool_call_id = elem.get("toolCallId")
                tool_name = elem.get("toolName")
                tool_argument = json.dumps(elem.get("args"), ensure_ascii=False)
                tool_calls.append(
                    ChoiceDeltaToolCall(
                        index=tool_call_index,
                        id=tool_call_id,
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=tool_name,
                            arguments=tool_argument,
                        ),
                    )
                )
                tool_call_index += 1
            elif t == "tool-result":
                blocks.append(TextBlock(text=elem.get("result")))

        additional_kwargs = {}
        if tool_calls:
            additional_kwargs = {"tool_calls": tool_calls}

        return ChatMessage(
            role=role,
            blocks=blocks,
            additional_kwargs=additional_kwargs,
        )


def convert_to_chat_messages(messages: List[dict]):
    ret_messages = []
    for message in messages:
        message_dict = to_chat_message(message)
        if isinstance(message_dict, list):
            ret_messages.extend(message_dict)
        else:
            ret_messages.append(message_dict)
    return ret_messages
