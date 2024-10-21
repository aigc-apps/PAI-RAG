from llama_index.core.tools import FunctionTool
from pai_rag.integrations.agent.pai.pai_agent import PaiAgent
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

agent = PaiAgent.from_tools(tools=tools, llm=fc_llm)


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
