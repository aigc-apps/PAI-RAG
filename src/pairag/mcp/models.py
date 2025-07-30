from typing import Union, List, Any, Optional
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


class ChatAgentRequest(BaseModel):
    model: str  # 模型名称
    messages: Union[List[Any], List[ChatCompletionMessageParam]]  # 上下文聊天
    stream: Optional[bool] = False  # 流式输出

    mcp_servers: Optional[List[str]] = []
    kb_ids: Optional[List[str]] = []
    enable_search: Optional[bool] = False
    enable_thinking: Optional[bool] = False
    enable_mcp: Optional[bool] = False
    max_steps: Optional[int] = None

    enable_attachments: Optional[bool] = False  # 是否启用附件功能

    # llm args
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"  # allow extra fields
