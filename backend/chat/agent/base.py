from typing import List
from chat.llm.llm_model import PaiLlm, ChatResponseGenerator, ChatCompletionToolParam
from chat.agent.state import AgentState
from llama_index.core.tools.function_tool import FunctionTool
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper


class BaseAgent:
    def __init__(
        self,
        prompt: str,
        llm: PaiLlm=None,
        tools: list[FunctionTool]=[],
        name: str=None,
    ):
        self.prompt = prompt
        self.tools = tools
        self.llm = llm
        self.name = name or self.__class__.__name__

    async def _run_async(self, state: AgentState):
        raise NotImplementedError

    async def invoke_llm_async(self, messages: List[dict], tools: List[ChatCompletionToolParam]= None) -> ChatResponseGenerator:
        assert self.llm is not None, "Agent {self.name} 执行错误: 没有找到大模型!"

        return await self.llm.astream(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        gen = self._run_async(state)
        return gen
