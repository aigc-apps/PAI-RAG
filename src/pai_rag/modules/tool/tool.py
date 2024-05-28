import logging
import os
from typing import Dict, List, Any
from llama_index.tools.google import GoogleSearchToolSpec
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.tool.load_and_search_tool_spec import LoadAndSearchToolSpec
from llama_index.core.tools import FunctionTool
from pai_rag.modules.tool.default_tool_description_template import *

logger = logging.getLogger(__name__)


class ToolModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        type = self.config["type"]
        logger.info(
            f"""
            [Parameters][Tool:FunctionTool]
                tools = {self.config.get("type", "Unknown tool")}
            """
        )
        tools = []
        if "googlewebsearch" in type:
            tools.extend(self.get_google_web_search_tool())

        if "calculator" in type:
            tools.extend(self.get_calculator_tool())

        return tools

    def get_google_web_search_tool(self):
        google_spec = GoogleSearchToolSpec(
            key=self.config.get("google_search_api", None),
            engine=self.config.get("google_search_engine", None),
            num=10,
        )
        return LoadAndSearchToolSpec.from_defaults(
            tool=google_spec.to_tool_list()[0],
            name="google_search",
            description=DEFAULT_GOOGLE_SEARCH_TOOL_DESP,
        ).to_tool_list()

    def get_calculator_tool(self):
        def multiply(a: int, b: int) -> int:
            """Multiply two integers and returns the result integer"""
            return a * b

        def add(a: int, b: int) -> int:
            """Add two integers and returns the result integer"""
            return a + b

        def divide(a: int, b: int) -> float:
            """Divide two integers and returns the result as a float"""
            if b == 0:
                raise ValueError("Cannot divide by zero.")
            return a / b

        def subtract(a: int, b: int) -> int:
            """Subtract the second integer from the first and returns the result integer"""
            return a - b

        multiply_tool = FunctionTool.from_defaults(
            fn=multiply,
            name="calculate_multiply",
            description=DEFAULT_CALCULATE_MULTIPLY,
        )
        add_tool = FunctionTool.from_defaults(
            fn=add, name="calculate_add", description=DEFAULT_CALCULATE_ADD
        )
        divide_tool = FunctionTool.from_defaults(
            fn=divide, name="calculate_divide", description=DEFAULT_CALCULATE_DIVIDE
        )
        subtract_tool = FunctionTool.from_defaults(
            fn=subtract,
            name="calculate_subtract",
            description=DEFAULT_CALCULATE_SUBTRACT,
        )
        return [multiply_tool, add_tool, divide_tool, subtract_tool]
