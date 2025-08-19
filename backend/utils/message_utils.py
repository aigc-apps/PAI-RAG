from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    MessageRole,
)


def to_chat_messages(
    message_dict: dict,
) -> List[ChatMessage] | ChatMessage:
    assert "role" in message_dict
    assert "content" in message_dict
    role = message_dict["role"]
    content = message_dict.get("content")
    role = message_dict.get("role")
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content")
    blocks = []
    if isinstance(content, str):
        return ChatMessage(role=role, content=content)
    else:
        # list
        chat_messages = []
        for elem in content:
            t = elem.get("type")
            if t == "text":
                blocks.append(TextBlock(text=elem.get("text")))
                chat_messages.append(
                    ChatMessage(
                        role=role,
                        blocks=blocks,
                    )
                )
            elif t == "image_url":
                img = elem.get("image_url").get("url")
                detail = elem.get("image_url").get("detail", "auto")
                if img.startswith("data:"):
                    blocks.append(ImageBlock(image=img, detail=detail))
                else:
                    blocks.append(ImageBlock(url=img, detail=detail))
                chat_messages.append(
                    ChatMessage(
                        role=role,
                        blocks=blocks,
                    )
                )
            elif t == "tool-call":
                chat_messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=elem.get("result", ""),
                    )
                )
        return chat_messages


def convert_to_chat_messages(messages: List[dict]):
    ret_messages = []
    for message in messages:
        message_dict = to_chat_messages(message)
        if isinstance(message_dict, list):
            ret_messages.extend(message_dict)
        else:
            ret_messages.append(message_dict)
    return ret_messages


def get_content_from_messages(contents: List[dict]) -> str:
    content_str = "\n".join(item["text"] for item in contents if "text" in item)
    return content_str
