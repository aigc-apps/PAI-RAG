from pydantic import BaseModel
from typing import List, Dict, Optional
from llama_index.core.schema import QueryBundle
from llama_index.core.base.llms.types import ChatMessage
from dataclasses import dataclass


class RagQuery(BaseModel):
    question: str  # 输入的问题
    chat_history: List[
        Dict[str, str]
    ] | None = None  # chat_history：用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间顺序排列。默认为空
    session_id: str | None = None  # 会话id，用于区分不同会话
    stream: bool | None = False  # 是否流式输出
    citation: bool | None = False  # 是否使用引用标签
    with_intent: bool | None = False  # 是否使用意图
    index_name: str | None = None  # 索引名称
    search_web: bool | None = False  # 是否搜索网页
    system_role_template: str | None = None  # system prompt模板
    custom_prompt_template: str | None = None  # custom prompt模板
    return_reference: bool | None = False  # 是否返回参考文档


class RetrievalQuery(BaseModel):
    question: str  # 检索问题
    index_name: str | None = None  # 检索目标索引名称


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
    new_query: str | None = None  # 改写生成的查询


class ChatCompletionRequest(BaseModel):
    model: str  # 模型名称
    messages: List[ChatMessage]  # 上下文聊天
    max_tokens: Optional[int] = 1024  # 最大输出长度
    temperature: Optional[float] = 0.1  # temperature
    stream: Optional[bool] = False  # 流式输出
    index_name: Optional[str] = None  # 索引名称
    search_web: Optional[bool] = False  # 搜索网络
    citation: Optional[bool] = False  # 生成引用

    # debug purpose
    force_search_web: Optional[bool] = False  # 始终执行搜索
    force_no_search: Optional[bool] = False  # 始终执行llm，不搜索知识库和网络
    force_search_knowledgebase: Optional[bool] = False  # 始终执行知识库搜索


@dataclass
class PaiQueryBundle(QueryBundle):
    stream: bool = False
    no_retrieval: bool = False
    citation: bool = False
    chat_messages_str: str = None
    need_web_search: bool = False
