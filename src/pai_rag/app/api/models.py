from enum import Enum
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
from llama_index.core.schema import QueryBundle
from llama_index.core.base.llms.types import ChatMessage
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore


class RagQuery(BaseModel):
    # 新版上下文聊天，传入messages则无需传入question和chat_history和session_id, 推荐传入messages
    messages: List[ChatMessage] = []

    question: str | None = None  # 输入的问题，即将obsolete
    chat_history: List[
        Dict[str, str]
    ] | None = (
        []
    )  # chat_history：用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间顺序排列。默认为空
    session_id: str | None = None  # 会话id，用于区分不同会话
    stream: bool | None = False  # 是否流式输出
    citation: bool | None = False  # 是否使用引用标签
    with_intent: bool | None = False  # 是否使用意图
    index_name: str | None = None  # 索引名称
    system_role_template: str | None = None  # system prompt模板
    custom_prompt_template: str | None = None  # custom prompt模板
    return_reference: bool | None = False  # 是否返回参考文档
    model: str | None = None  # 推理模型
    temperature: float | None = 0.1  # 推理时参数：温度值

    # adapt to chat_request
    chat_knowledgebase: Optional[bool] = False  # 查询知识库
    search_web: Optional[bool] = False  # 搜索网络
    chat_llm: Optional[bool] = False  # 是否使用llm聊天
    chat_agent: Optional[bool] = False  # 是否使用agent
    chat_db: Optional[bool] = False  # 查询数据库
    chat_news: Optional[bool] = False  # 使用新闻工具


class ContextDoc(BaseModel):
    text: str  # 文档文本
    score: float  # 文档得分
    metadata: Dict  # 文档元数据
    image_url: str | None = None  # 图片链接


class RetrievalResponse(BaseModel):
    docs: List[ContextDoc]


class RagResponse(BaseModel):
    answer: str  # 答案
    session_id: str | None = None  # 会话id，用于区分不同会话
    docs: List[ContextDoc] | None = None  # 搜索到的文档


class ChatCompletionRequest(BaseModel):
    model: str  # 模型名称
    messages: List[ChatMessage]  # 上下文聊天
    max_tokens: Optional[int] = 1024  # 最大输出长度
    temperature: Optional[float] = 0.1  # temperature
    stream: Optional[bool] = False  # 流式输出
    index_name: Optional[str] = None  # 索引名称
    chat_knowledgebase: Optional[bool] = False  # 查询知识库
    search_web: Optional[bool] = False  # 搜索网络
    citation: Optional[bool] = False  # 生成引用
    return_reference: Optional[bool] = False  # 返回参考
    chat_llm: Optional[bool] = False  # llm聊天
    chat_agent: Optional[bool] = False  # 使用agent
    chat_db: Optional[bool] = False  # 查询数据库
    chat_news: Optional[bool] = False  # 使用新闻工具

    # llm args
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatToolType(str, Enum):
    SEARCH_WEB = "search_web"
    CHAT_NEWS = "chat_news"
    CHAT_KNOWLEDGEBASE = "chat_knowledgebase"
    CHAT_DB = "chat_db"
    CHAT_AGENT = "chat_agent"
    CHAT_LLM = "chat_llm"


class ChatIntentType(str, Enum):
    SEARCH_WEB = "search_web"  # search web
    CHAT_LLM = "chat_llm"  # llm chat
    LIST_NEWS = "list_news"  # list news
    CHAT_NEWS = "chat_news"  # chat news
    CHAT_NEWS_LLM = "chat_news_llm"  # chat news only by llm
    CHAT_KNOWLEDGEBASE = "chat_knowledgebase"
    CHAT_AGENT = "chat_agent"  # chat agent
    CHAT_DB = "chat_db"  # chat sql


@dataclass
class PaiQueryBundle(QueryBundle):
    system_role: str | None = None
    messages: Optional[List[ChatMessage]] = None
    stream: bool = False
    intent: ChatIntentType = ChatIntentType.CHAT_KNOWLEDGEBASE
    no_retrieval: bool = False
    citation: bool = False
    original_query_str: str = None
    chat_messages_str: str = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    llm_kwargs: Optional[Dict[str, Any]] = None
    model: str | None = None
    news_topics: Optional[List[str]] = None


class ChatResponseWrapper(BaseModel):
    response: Any
    additional_kwargs: Dict[str, Any] = {}
    source_nodes: List[NodeWithScore] = []
