# chat/agent/reactor.py

import json
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from chat.agent.base import BaseAgent
from chat.agent.state import AgentState  # 如果需要，也可移除依赖
from llama_index.core.tools.function_tool import FunctionTool, ToolOutput
from chat.llm.llm_model import PaiLlm
from chat.llm.models import TextChunk, ChatResponseGenerator, ToolResultChunk
from extensions.trace.base import use_current_span

from opentelemetry import trace


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)


MAX_RECURSION_STEPS = 20



class Actor(BaseAgent):
    def __init__(
        self,
        prompt: str,
        llm: PaiLlm,
        tools: list[FunctionTool],
        name: str = "actor",
        max_steps: int = MAX_RECURSION_STEPS
    ):
        super().__init__(prompt, llm, tools, name)
        self.max_steps = max_steps
        self.tool_fn_map = {tool.metadata.name: tool for tool in self.tools}
        self.tool_metadata = [
            tool.metadata.to_openai_tool() for tool in self.tools
        ]

    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        """
        执行 ReAct 循环：持续调用工具直到 LLM 不再返回 tool_calls 或达到最大步数。
        :param initial_messages: 初始消息列表（包含 system + 历史 + 第一次 tool_call 和结果）
        :return: 异步生成器，yield 所有中间 chunk
        """
        logger.info(f"[{self.name}] Starting ReAct loop.")

        messages = [{"role": "system", "content": self.prompt}] + state.messages

        @use_current_span(trace.get_current_span())
        async def gen():
            react_step = 1

            while react_step <= self.max_steps:
                logger.info(f"[{self.name}] ReAct step {react_step} / {self.max_steps}")
                react_step += 1

                tool_calls = []
                next_content = ""

                # 调用 LLM 获取下一步决策
                async for chunk in await self.invoke_llm_async(
                    messages=messages,
                    tools=self.tool_metadata,
                ):
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                    if chunk.delta:
                        next_content += chunk.delta
                        yield TextChunk(delta=chunk.delta)

                if next_content:
                    messages.append({
                        "role": "assistant",
                        "content": next_content,
                    })

                if not tool_calls:
                    logger.info(f"[{self.name}] No more tool calls. Exiting ReAct loop.")
                    break

                # 处理每个工具调用
                for tool in tool_calls:
                    if tool.type != "function":
                        continue

                    function_name = tool.function.name
                    if not function_name or function_name not in self.tool_fn_map:
                        logger.warning(f"[{self.name}] Unknown tool: {function_name}, skipping.")
                        continue

                    # 若调用 respond-tool，主动结束
                    if function_name == "respond-tool":
                        logger.info(f"[{self.name}] respond-tool called. Exiting.")
                        yield TextChunk(tool_calls=[tool])
                        return

                    # 解析参数
                    try:
                        function_args = json.loads(tool.function.arguments) if tool.function.arguments else {}
                    except json.JSONDecodeError:
                        logger.error(f"[{self.name}] Invalid JSON args: {tool.function.arguments}")
                        function_args = {}

                    yield TextChunk(tool_calls=[tool])

                    # 调用工具
                    async_fn = self.tool_fn_map[function_name]
                    logger.info(f"[{self.name}] Calling {function_name} with args: {function_args}")
                    tool_result = await call_tool_with_retry(async_fn, function_args)
                    logger.info(f"[{self.name}] Tool result: {tool_result.content[:200]}...")

                    # 追加 assistant 消息（带 tool_calls）
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool]
                    })

                    # 追加工具执行结果
                    messages.append({
                        "role": "tool",
                        "content": tool_result.content,
                        "tool_call_id": tool.id
                    })

                    yield ToolResultChunk(
                        tool=tool,
                        result=tool_result.content,
                    )

            # 超出步数保护
            if react_step > self.max_steps:
                yield TextChunk(delta="任务失败: 超出最大迭代次数，任务已结束。")
                return

        return gen()
