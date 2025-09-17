
from typing import Annotated
from llama_index.core.tools.function_tool import FunctionTool
from loguru import logger


PLAN_TOOL_DESCRIPTION = """
Break down user requests into clear, sequential steps that can be executed by worker agents. Each step must be a single, unambiguous action.

## Guidelines
- **Atomic & Independent**: Each step must represent a single, executable action. No compound tasks.
- **Self-Contained**: Include all necessary context: who, what, where, and when — derived from available information.
- **Action-Oriented Language**: Start with strong verbs (e.g., "Verify", "Retrieve", "Generate"). Avoid vague terms like "check" or "review" without specifics.
- **Measurable Outcome**: Each step must produce a concrete, observable result that can be validated.
- **Concise Sequence**: Limit plans to 3-5 essential steps. Omit redundant or implied actions.
- **Valid JSON Only**: Return a well-formed JSON object. No additional text, explanation, or markdown.
- **Language Consistency**: Use the same language as the user's query unless specified otherwise.

"""

async def aget_plan_tool():
    async def plan_func(
        steps: Annotated[
            list[str],
            "List of plan steps. 针对当前任务拟定的计划或方案，将复杂问题分解为多个可执行步骤。",
        ] = "",
    ):
        logger.info(
            f"Plan: {steps}"
        )
        return steps

    plan_tool = FunctionTool.from_defaults(
        async_fn=plan_func,
        name="planning-tool",
        description=PLAN_TOOL_DESCRIPTION,
    )

    return plan_tool


RESPONSE_TOOL_DESCRIPTION = """
Used in stage when no additional tools are needed/available.
Generate final response to the user directly based on the context and information you have.
"""

async def aget_respond_tool():
    async def response_func():
        logger.info(
            "Plan completed. Generating response..."
        )
        return

    response_tool = FunctionTool.from_defaults(
        async_fn=response_func,
        name="respond-tool",
        description=RESPONSE_TOOL_DESCRIPTION,
    )

    return response_tool
