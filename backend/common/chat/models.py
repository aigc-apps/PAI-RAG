from typing import Dict, Literal, Union, List, Any, Optional, Sequence
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field


DEFAULT_GUARDRAIL_ADVICE = "作为人工智能助手，我无法回应包含不当或敏感信息的内容。"
SupportedComparisonOperator = Literal[
    # for string or array
    "contains",
    "not contains",
    "start with",
    "end with",
    "is",
    "is not",
    "empty",
    "not empty",
    # for number
    "=",
    "≠",
    ">",
    "<",
    "≥",
    "≤",
    # for time
    "before",
    "after",
]


class Condition(BaseModel):
    """
    Condition detail
    """

    name: str
    comparison_operator: SupportedComparisonOperator
    value: str | Sequence[str] | None | int | float = None


class MetadataFilteringCondition(BaseModel):
    """
    Metadata Filtering Condition.
    """

    logical_operator: Optional[Literal["and", "or"]] = "and"
    conditions: Optional[list[Condition]] = Field(default=None, deprecated=True)


class DocRecord(BaseModel):
    content: str  # 包含知识库中数据源的文本块
    score: float  # 结果与查询的相关性分数，范围：0~1
    title: str  # 文档标题
    metadata: Dict  # 包含数据源中文档的元数据属性及其值


class NewRetrievalResponse(BaseModel):
    records: List[DocRecord]


class RetrievalSetting(BaseModel):
    top_k: Optional[int] = None
    score_threshold: Optional[float] = 0.4


class RetrievalRequest(BaseModel):
    knowledge_id: Optional[str] = "default"  # 知识库名称（index_name）
    query: str  # 查询内容
    user_id: Optional[str] = None
    retrieval_setting: Optional[RetrievalSetting] = None
    metadata_condition: Optional[MetadataFilteringCondition] = None
    # ["retrieval_mode", "similarity_top_k", "vector_weight", "keyword_weight", "reranker_type", "similarity_threshold", "reranker_similarity_threshold", "reranker_model", "reranker_similarity_top_k"]


class ChatAgentRequest(BaseModel):
    model: str  # 模型名称
    messages: Union[List[Any], List[ChatCompletionMessageParam]]  # 上下文聊天
    stream: Optional[bool] = False  # 默认流式输出

    mcp_ids: Optional[List[str]] = []
    kb_ids: Optional[List[str]] = []
    enable_search: Optional[bool] = False
    enable_agent: Optional[bool] = False
    max_steps: Optional[int] = None
    user_id: Optional[str] = None
    enable_input_guardrail: Optional[bool] = False
    enable_output_guardrail: Optional[bool] = False
    guardrail_hint: Optional[str] = DEFAULT_GUARDRAIL_ADVICE

    # llm args
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"  # allow extra fields
