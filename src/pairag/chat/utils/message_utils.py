from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    MessageRole,
)
import re
from typing import List, TypedDict, Required, Literal
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)
from openai.types.chat import ChatCompletionUserMessageParam
from pairag.integrations.llms.utils.utils import extract_image_links


def from_openai_message_dict(message_dict: dict) -> ChatMessage:
    """Convert openai message dict to generic message."""
    role = message_dict.get("role")
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content")
    blocks = []
    if isinstance(content, list):
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
            else:
                msg = f"Unsupported message type: {t}"
                raise ValueError(msg)
        content = None

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return ChatMessage(
        role=role, content=content, additional_kwargs=additional_kwargs, blocks=blocks
    )


def extract_openai_message_content(message_dict: dict) -> str:
    """Extract content from openai message dict."""
    content = message_dict.get("content")
    if isinstance(content, list):
        content_list = []
        for elem in content:
            t = elem.get("type")
            if t == "text":
                content_list.append(elem.get("text"))
            elif t == "image_url":
                content_list.append(elem.get("image_url").get("url"))
            else:
                msg = f"Unsupported message type: {t}"
                raise ValueError(msg)
        content = "\n".join(content_list)
        return content
    else:
        return content


def remove_think_from_messages(messages: List[ChatMessage]):
    for message in messages:
        new_blocks = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                # 对文本内容进行正则替换
                cleaned_text = re.sub(r"</think>\n*", "", block.text, flags=re.DOTALL)
                if cleaned_text.strip():
                    new_blocks.append(TextBlock(text=cleaned_text))
            else:
                new_blocks.append(block)
        message.blocks = new_blocks
    return messages


def message_is_empty(messages: List[ChatMessage]):
    if len(messages) == 0 or messages[-1].content is None or messages[-1].content == "":
        return True

    return False


class ImageURL(TypedDict, total=False):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image.

    Learn more in the
    [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
    """


def parse_system_prompt(messages: List[ChatCompletionUserMessageParam]):
    messages = [message for message in messages if message["content"]]
    if len(messages) > 0 and messages[0]["role"] == MessageRole.SYSTEM:
        system_prompt = messages[0]["content"]
        return system_prompt, messages[1:]

    return None, messages


def parse_messages(
    messages: List[ChatCompletionUserMessageParam],
) -> List[ChatMessage]:
    num_messages = len(messages)
    chat_messages = []
    for index, message in enumerate(messages):
        if index == num_messages - 1 and isinstance(message["content"], str):
            image_list = extract_image_links(message["content"])
            if image_list:
                image_josn_list = [
                    ChatCompletionContentPartImageParam(
                        type="image_url", image_url=ImageURL(url=image_url)
                    )
                    for image_url in image_list
                ]
                content = [
                    ChatCompletionContentPartTextParam(
                        type="text", text=message["content"]
                    )
                ]
                content.extend(image_josn_list)
                message = {"role": message["role"], "content": content}
        chat_messages.append(from_openai_message_dict(message))

    return chat_messages
