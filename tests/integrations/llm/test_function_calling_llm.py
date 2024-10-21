from llama_index.core.tools import FunctionTool
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.llm_config import DashScopeLlmConfig

fc_llm_config = DashScopeLlmConfig(model="qwen-max")
fc_llm = PaiLlm(fc_llm_config)


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

tools = [multiply_tool, add_tool]


def test_fc_llm_chat_with_tools():
    response = fc_llm.chat_with_tools(tools=tools, user_msg="What is (121 + 2) * 5?")
    tool_calls = fc_llm.get_tool_calls_from_response(
        response, error_on_no_tool_call=False
    )
    assert len(tool_calls) > 0
    for _, tool_call in enumerate(tool_calls):
        if tool_call.tool_name == "add":
            assert tool_call.tool_kwargs["a"] == 121
            assert tool_call.tool_kwargs["b"] == 2
        if tool_call.tool_name == "multiply":
            assert tool_call.tool_kwargs["a"] == 123
            assert tool_call.tool_kwargs["b"] == 5
