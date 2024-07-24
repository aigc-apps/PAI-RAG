from llama_index.tools.google import GoogleSearchToolSpec
from pai_rag.modules.tool.load_and_search_tool_spec import LoadAndSearchToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.bridge.pydantic import FieldInfo, create_model
from pai_rag.modules.tool.sample_tools_for_booking import function_mapper
import json
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


def get_booking_demo_tool():
    booking_tools = []

    # 加载JSON文件
    with open("example_data/sample_tools_for_booking.json", "r") as file:
        custom_tools = json.load(file)

        for c_tool in custom_tools:
            fn_name = c_tool["function"]["name"]

            fn_schema = create_tool_fn_schema(fn_name, c_tool["function"]["parameters"])
            tool = FunctionTool.from_defaults(
                fn=function_mapper[fn_name],
                name=fn_name,
                fn_schema=fn_schema,
                description=c_tool["function"]["description"],
            )
            booking_tools.append(tool)
    return booking_tools


def get_customized_tool(config):
    return []
