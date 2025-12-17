from typing import List, Callable, Awaitable, Optional
from common.llm.llm_model import PaiLlm, ChatResponseGenerator, ChatCompletionToolParam
from agent.state import AgentState
from llama_index.core.tools.function_tool import FunctionTool
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from loguru import logger


class BaseAgent:
    def __init__(
        self,
        prompt: str,
        llm: PaiLlm=None,
        tools: list[FunctionTool]=[],
        name: str=None,
        cleanup_func: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.prompt = prompt
        self.tools = tools
        self.llm = llm
        self.name = name or self.__class__.__name__
        self._cleanup_func = cleanup_func
        self._cleanup_called = False

    async def invoke_llm_async(self, messages: List[dict], tools: List[ChatCompletionToolParam]= None) -> ChatResponseGenerator:
        assert self.llm is not None, "Agent {self.name} 执行错误: 没有找到大模型!"

        return await self.llm.astream(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    def with_cleanup(self, gen_func):
        """装饰器：在异步生成器函数完成后自动调用 cleanup 函数"""
        async def wrapped_gen(*args, **kwargs):
            response_gen = gen_func(*args, **kwargs)
            if not self._cleanup_func:
                async for item in response_gen:
                    yield item
                return

            try:
                async for item in response_gen:
                    yield item
            finally:
                # 确保清理函数只调用一次
                if not self._cleanup_called:
                    self._cleanup_called = True
                    try:
                        await self._cleanup_func()
                    except Exception as e:
                        logger.exception(f"Failed to cleanup in {self.name}: {e}")

        return wrapped_gen

    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        raise NotImplementedError
