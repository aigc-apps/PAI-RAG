from typing import List, Dict
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from pai_rag.integrations.chat_store.pai.pai_chat_store import PaiChatStore


# 解析函数，将List[Dict[str, str]]转换为List[ChatMessage]
def parse_chat_messages(raw_data: List[Dict[str, str]]) -> List[ChatMessage]:
    chat_messages = []
    for pair in raw_data:
        # 假设Dict的第一个元素是user的消息，第二个是assistant的消息
        user_message = ChatMessage(role=MessageRole.USER, content=pair["user"])
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=pair["bot"])
        chat_messages.extend([user_message, assistant_message])  # 添加到列表
    return chat_messages


# 兼容旧接口
# 当有chat_history参数时，message=chat_history+question
# 否则，message=session_history+question
def parse_chat_messages_v2(
    question: str,
    session_id: str,
    chat_history: List[Dict[str, str]],
    chat_store: PaiChatStore,
):
    messages = []
    if len(chat_history) > 0:
        messages.extend(parse_chat_messages(chat_history))
    else:
        messages.extend(chat_store.get_messages(session_id))

    messages.append(ChatMessage(role=MessageRole.USER, content=question))
    return messages
