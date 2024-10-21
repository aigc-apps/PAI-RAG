from typing import Dict, List
from pydantic import BaseModel
from openai.types.beta.function_tool import FunctionTool


class ApiParameter(BaseModel):
    type: str
    description: str


class ApiTool(BaseModel):
    name: str
    url: str
    headers: Dict[str, str]
    method: str
    description: str
    content_type: str | None = None
    parameters: Dict[str, ApiParameter]
    required: List[str]


class PaiAgentToolDefinition(BaseModel):
    intents: Dict[str, str] = {}
    system_prompt: str | None = None
    function_tools: List[FunctionTool] = []
    api_tools: List[ApiTool] = []
