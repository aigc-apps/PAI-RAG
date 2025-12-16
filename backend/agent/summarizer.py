from common.llm.llm_model import ChatResponseGenerator
from common.chat.constants import MessageRole
from agent.base import BaseAgent
from agent.state import AgentState
from agent.prompts import SUMMARY_PROMPT
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from extensions.trace.base import use_current_span
from loguru import logger
from opentelemetry import trace


def build_synthesize_prompt(state: AgentState, prompt: str):
    current_datetime = state.context_variables.get("current_datetime", "")
    prompt = SUMMARY_PROMPT.format(
        tool_results=state.observations,
        chat_history=state.chat_history,
        current_datetime=current_datetime,
        user_query=state.user_query,
    )
    return prompt


class Summarizer(BaseAgent):

    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        logger.info("Running summarizer agent.")
        try:
            prompt = build_synthesize_prompt(
                state=state,
                prompt=self.prompt,
            )

            @use_current_span(trace.get_current_span())
            async def gen():
                response_gen = await self.invoke_llm_async(messages=[
                    {"role": MessageRole.USER, "content": prompt},
                ])
                async for chunk in response_gen:
                    yield chunk

            return self._wrap_generator_with_cleanup(gen())
        except Exception:
            # 如果执行出错，也要清理
            if not self._cleanup_called and self._cleanup_func:
                self._cleanup_called = True
                try:
                    await self._cleanup_func()
                except Exception as cleanup_error:
                    logger.exception(f"Failed to cleanup in {self.name} after error: {cleanup_error}")
            raise
