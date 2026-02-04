from typing import List
from pydantic import BaseModel, Field
from common.chat.constants import MessageRole


def get_message_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        text = ""
        for item in content:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text

    return content


class AgentState(BaseModel):
    messages: List[dict]
    step: int = Field(default=1)

    @classmethod
    def from_messages(
        cls,
        messages: List[dict],
    ):
        filtered_messages = []
        for message in messages:
            content = get_message_content(message)
            if message["role"] == MessageRole.USER:
                filtered_messages.append(message)
            elif message["role"] == MessageRole.ASSISTANT:
                msg_content = message.get("content", "")
                if isinstance(msg_content, list):
                    content = []
                    for item in msg_content:
                        if item.get("type") == "text":
                            content.append(item)

                    if len(content) > 0:
                        filtered_messages.append({"role": MessageRole.ASSISTANT, "content": content})

                else:
                    filtered_messages.append({"role": MessageRole.ASSISTANT, "content": msg_content})

        return cls(
            messages=filtered_messages,
            step=1,
        )
