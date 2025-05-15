from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type
import json


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


class ToolOutput(BaseModel):
    """Tool output."""

    content: str
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any
    is_error: bool = False

    def __str__(self) -> str:
        """String."""
        return str(self.content)


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema
    return_direct: bool = False

    def get_parameters_dict(self) -> dict:
        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = self.fn_schema.model_json_schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions", "$defs"]
            }
        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters, ensure_ascii=False)

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        """To OpenAI tool."""
        if not skip_length_check and len(self.description) > 1024:
            raise ValueError(
                "Tool description exceeds maximum length of 1024 characters. "
                "Please shorten your description or move it to the prompt."
            )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_dict(),
            },
        }
