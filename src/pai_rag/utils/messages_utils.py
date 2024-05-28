from typing import List, Dict
from llama_index.core.base.llms.types import ChatMessage, MessageRole


# 解析函数，将List[Dict[str, str]]转换为List[ChatMessage]
def parse_chat_messages(raw_data: List[Dict[str, str]]) -> List[ChatMessage]:
    chat_messages = []
    for pair in raw_data:
        # 假设Dict的第一个元素是user的消息，第二个是assistant的消息
        user_message = ChatMessage(role=MessageRole.USER, content=pair["user"])
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=pair["bot"])
        chat_messages.extend([user_message, assistant_message])  # 添加到列表
    return chat_messages
