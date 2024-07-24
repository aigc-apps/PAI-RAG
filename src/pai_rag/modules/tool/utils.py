from llama_index.tools.google import GoogleSearchToolSpec
from pai_rag.modules.tool.load_and_search_tool_spec import LoadAndSearchToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.bridge.pydantic import FieldInfo, create_model
import json
import os
import sys
from pai_rag.modules.tool.default_tool_description_template import (
    DEFAULT_GOOGLE_SEARCH_TOOL_DESP,
    DEFAULT_CALCULATE_MULTIPLY,
    DEFAULT_CALCULATE_ADD,
    DEFAULT_CALCULATE_DIVIDE,
    DEFAULT_CALCULATE_SUBTRACT,
)


def create_tool_fn_schema(name, params):
    fields = {}
    params_prop = params["properties"]
    for param_name in params_prop:
        param_type = params_prop[param_name]["type"]
        param_desc = params_prop[param_name]["description"]
        fields[param_name] = (param_type, FieldInfo(description=param_desc))
    return create_model(name, **fields)  # type: ignore


def get_google_web_search_tool(config):
    google_spec = GoogleSearchToolSpec(
        key=config.get("google_search_api", None),
        engine=config.get("google_search_engine", None),
        num=10,
    )
    return LoadAndSearchToolSpec.from_defaults(
        tool=google_spec.to_tool_list()[0],
        name="google_search",
        description=DEFAULT_GOOGLE_SEARCH_TOOL_DESP,
    ).to_tool_list()


def get_calculator_tool():
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


def get_customized_tool(config):
    func_path = config["func_path"]
    sys.path.append(func_path)
    module = __import__("custom_functions")
    tools = []
    # 加载JSON文件
    with open(os.path.join(func_path, "custom_functions.json"), "r") as file:
        custom_tools = json.load(file)

        for c_tool in custom_tools:
            fn_name = c_tool["function"]["name"]
            if hasattr(module, fn_name):
                func = getattr(module, fn_name)
                fn_schema = create_tool_fn_schema(
                    fn_name, c_tool["function"]["parameters"]
                )
                tool = FunctionTool.from_defaults(
                    fn=func,
                    name=fn_name,
                    fn_schema=fn_schema,
                    description=c_tool["function"]["description"],
                )
                tools.append(tool)
            else:
                raise ValueError(
                    f"Function {fn_name} has not been defined in the custom_functions.py, please define it."
                )
    return tools
