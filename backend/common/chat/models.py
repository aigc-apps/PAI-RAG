from typing import Dict, Literal, Union, List, Any, Optional, Sequence
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, model_validator
from common.knowledgebase.types import VectorIndexRetrievalType


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
    # for list
    "in",
    "not in",
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

    @model_validator(mode="after")
    def validate_in_operator_value(self):
        if self.comparison_operator in ("in", "not in"):
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(
                    f"Operator '{self.comparison_operator}' requires a list value, got {type(self.value).__name__}"
                )
        return self


class MetadataFilteringCondition(BaseModel):
    """
    Metadata Filtering Condition.
    Supports nested structure via condition_groups for complex queries like:
    (category = 'COMMON' OR category = 'PC') AND language = 'en-US'
    """

    logical_operator: Optional[Literal["and", "or"]] = "and"
    conditions: Optional[list[Condition]] = Field(default=None, deprecated=True)
    condition_groups: Optional[list["MetadataFilteringCondition"]] = None

    @staticmethod
    def merge(
        base: Optional["MetadataFilteringCondition"],
        override: Optional["MetadataFilteringCondition"],
    ) -> Optional["MetadataFilteringCondition"]:
        """Merge two MetadataFilteringCondition with AND logic.

        base: user-provided hard constraint (always applied)
        override: agent-generated supplementary filter
        """
        if not base:
            return override
        if not override:
            return base
        return MetadataFilteringCondition(
            logical_operator="and",
            condition_groups=[base, override],
        )


class DocRecord(BaseModel):
    content: str  # 包含知识库中数据源的文本块
    score: float  # 结果与查询的相关性分数，范围：0~1
    title: str  # 文档标题
    metadata: Dict  # 包含数据源中文档的元数据属性及其值
    url: Optional[str] = None  # 文档的URL


class NewRetrievalResponse(BaseModel):
    records: List[DocRecord]


class RetrievalSetting(BaseModel):
    retrieval_mode: Optional[VectorIndexRetrievalType] = None
    vector_weight: Optional[float] = None
    enable_rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_provider_name: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    score_threshold: Optional[float] = None # For dify compatibility, similarity_threshold is preferred
    rerank_top_k: Optional[int] = None

    @model_validator(mode="after")
    def validate_score_threshold(self):
        if self.similarity_threshold is None:
            self.similarity_threshold = self.score_threshold
        return self


class RetrievalRequest(BaseModel):
    knowledge_id: Optional[str] = None  # 知识库ID（knowledge_id）
    knowledge_id_list: Optional[List[str]] = None # 知识库ID列表

    query: str  # 查询内容
    user_id: Optional[str] = None
    retrieval_setting: Optional[RetrievalSetting] = None
    metadata_condition: Optional[MetadataFilteringCondition] = None
    # ["retrieval_mode", "similarity_top_k", "vector_weight", "keyword_weight", "reranker_type", "similarity_threshold", "reranker_similarity_threshold", "reranker_model", "reranker_similarity_top_k"]


class ChatAgentRequest(BaseModel):
    model: str  # 模型名称
    messages: Union[List[Any], List[ChatCompletionMessageParam]]  # 上下文聊天
    stream: Optional[bool] = True  # 默认流式输出

    mcp_ids: Optional[List[str]] = []
    kb_ids: Optional[List[str]] = []
    faq_config: Optional[dict] = None
    enable_faq: Optional[bool] = False
    enable_search: Optional[bool] = False
    enable_agent: Optional[bool] = False
    enable_chatdb: Optional[bool] = False
    vision_model_id: Optional[str] = None
    max_steps: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None  # 会话ID，用于会话历史管理
    enable_input_guardrail: Optional[bool] = False
    enable_output_guardrail: Optional[bool] = False
    guardrail_hint: Optional[str] = DEFAULT_GUARDRAIL_ADVICE
    metadata_condition: Optional[MetadataFilteringCondition] = None
    enable_auto_metadata_filter: Optional[bool] = False

    # llm args
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    prompts: Optional[Dict[str, str]] = None

    class Config:
        extra = "allow"  # allow extra fields
