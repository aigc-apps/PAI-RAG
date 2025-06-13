from pydantic import BaseModel
from typing import Any, List, Dict, Optional, AsyncGenerator, Generator, Union
from openai.types.chat import ChatCompletionMessageParam
from llama_index.core.schema import NodeWithScore
from pairag.integrations.query_transform.pai_query_transform import IntentResult


class ContextDoc(BaseModel):
    text: str  # 文档文本
    score: float  # 文档得分
    metadata: Dict  # 文档元数据
    image_url: str | None = None  # 图片链接


class RetrievalRequest(BaseModel):
    knowledgebase_id: Optional[str] = "default"  # 知识库名称（index_name）
    query: str  # 查询内容
    retrieval_settings: Optional[Dict] = None
    # ["retrieval_mode", "similarity_top_k", "vector_weight", "keyword_weight", "reranker_type", "similarity_threshold", "reranker_similarity_threshold", "reranker_model", "reranker_similarity_top_k"]


class DocRecord(BaseModel):
    content: str  # 包含知识库中数据源的文本块
    score: float  # 结果与查询的相关性分数，范围：0~1
    title: str  # 文档标题
    metadata: Dict  # 包含数据源中文档的元数据属性及其值


class NewRetrievalResponse(BaseModel):
    records: List[DocRecord]


class RetrievalResponse(BaseModel):
    docs: List[ContextDoc]


class ChatCompletionRequest(BaseModel):
    model: str  # 模型名称
    messages: Union[List[Any], List[ChatCompletionMessageParam]]  # 上下文聊天
    stream: Optional[bool] = False  # 流式输出
    index_name: Optional[str] = None  # 索引名称
    chat_knowledgebase: Optional[bool] = False  # 查询知识库
    search_web: Optional[bool] = False  # 搜索网络
    return_reference: Optional[bool] = False  # 返回参考
    chat_llm: Optional[bool] = False  # llm聊天
    chat_db: Optional[bool] = False  # 查询数据库
    chat_news: Optional[bool] = False  # 使用新闻工具
    # llm args
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    intent: Optional[IntentResult] = None  # 意图

    class Config:
        extra = "allow"  # allow extra fields


class ChatResponseWrapper(BaseModel):
    response: Any
    additional_kwargs: Dict[str, Any] = {}
    source_nodes: List[NodeWithScore] = []
    intent_result: Optional[IntentResult] = None

    def model_dump_json(self, exclude=None, **kwargs) -> str:
        if exclude is None:
            exclude = set()
        elif isinstance(exclude, dict):
            exclude = {k for k, v in exclude.items() if v}

        # to compatible with arize instrumentation
        if isinstance(self.response, (Generator, AsyncGenerator)):
            exclude.add("response")

        return super().model_dump_json(exclude=exclude, **kwargs)


class EmbeddingInput(BaseModel):
    input: str | List[str] = None
    model: str = "bge-m3"
