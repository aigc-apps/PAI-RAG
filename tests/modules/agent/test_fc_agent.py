import os
from pai_rag.integrations.agent.function_calling.step import (
    MyFunctionCallingAgentWorker,
)
from pai_rag.integrations.llms.dashscope.fc_base import MyFCDashScope
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

tools = [multiply_tool, add_tool]

fc_llm = MyFCDashScope(
    model_name="qwen2-7b-instruct", api_key=os.getenv("DASHSCOPE_API_KEY")
)
fc_agent_worker = MyFunctionCallingAgentWorker.from_tools(
    tools=tools,
    llm=fc_llm,
    system_prompt="你是一个调用计算插件的计算小助手，请严格使用已知的插件来计算，不允许虚构。",
    verbose=True,
)
agent = fc_agent_worker.as_agent()


def test_fc_agent_chat():
    response = agent.chat("What is (121 + 2) * 5?")
    assert len(response.sources) == 2
    for i, tool_call in enumerate(response.sources):
        if i == 0:
            # content='123', tool_name='add', raw_input={'args': (), 'kwargs': {'a': 121, 'b': 2}
            assert tool_call.content == "123"
            assert tool_call.tool_name == "add"
            assert tool_call.raw_input["kwargs"]["a"] == 121
            assert tool_call.raw_input["kwargs"]["b"] == 2
        if i == 1:
            # content='615', tool_name='multiply', raw_input={'args': (), 'kwargs': {'a': 123, 'b': 5}
            assert tool_call.content == "615"
            assert tool_call.tool_name == "multiply"
            assert tool_call.raw_input["kwargs"]["a"] == 123
            assert tool_call.raw_input["kwargs"]["b"] == 5
