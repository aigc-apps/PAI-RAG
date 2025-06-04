from enum import Enum
from pydantic import BaseModel
from openai.types.completion_usage import CompletionUsage


class ChatToolType(str, Enum):
    SEARCH_WEB = "search_web"
    CHAT_NEWS = "chat_news"
    CHAT_KNOWLEDGEBASE = "chat_knowledgebase"
    CHAT_DB = "chat_db"
    CHAT_LLM = "chat_llm"


class ChatIntentType(str, Enum):
    SEARCH_WEB = "search_web"  # search web
    CHAT_LLM = "chat_llm"  # llm chat
    CHAT_NEWS = "chat_news"  # chat news
    CHAT_NEWS_LLM = "chat_news_llm"  # chat news only by llm
    CHAT_KNOWLEDGEBASE = "chat_knowledgebase"
    CHAT_DB = "chat_db"  # chat sql


class IntentResult(BaseModel):
    intent: ChatIntentType = ChatIntentType.CHAT_LLM
    query_str: str = None
    token_usage: CompletionUsage = CompletionUsage(
        completion_tokens=0, prompt_tokens=0, total_tokens=0
    )
