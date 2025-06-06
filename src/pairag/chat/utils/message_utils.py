from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    TextBlock,
)


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
