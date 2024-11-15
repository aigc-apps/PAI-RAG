from typing import Dict, List, Self
from pydantic import BaseModel, model_validator
from openai.types.beta.function_tool import FunctionTool
from loguru import logger

DEFAULT_TOOL_DEFINITION_FILE = "./example_data/function_tools/api-tool-with-intent-detection-for-travel-assistant/tools.json"
DEFAULT_PYTHONSCRIPT_FILE = "./example_data/function_tools/api-tool-with-intent-detection-for-travel-assistant/tools.py"


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


class AgentConfig(BaseModel):
    system_prompt: str | None = None
    python_scripts: str | None = None
    function_definition: str | None = None
    api_definition: str | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        if not self.api_definition and not self.function_definition:
            logger.info("No agent tool definition found. Will use demo tools.")
            with open(DEFAULT_PYTHONSCRIPT_FILE) as py_file:
                self.python_scripts = py_file.read()
            with open(DEFAULT_TOOL_DEFINITION_FILE) as tool_file:
                import json

                json_config = json.loads(tool_file.read())
                self.system_prompt = json_config["system_prompt"]
                self.api_definition = json.dumps(
                    json_config["api_tools"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                self.function_definition = json.dumps(
                    json_config["function_tools"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )

        return self


class PaiAgentDefinition(BaseModel):
    system_prompt: str | None = None
    python_scripts: str | None = None
    function_tools: List[FunctionTool] = []
    api_tools: List[ApiTool] = []
