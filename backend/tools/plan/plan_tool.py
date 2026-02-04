
from typing import Annotated
from llama_index.core.tools.function_tool import FunctionTool
import json

from loguru import logger


PLAN_TOOL_DESCRIPTION = """
Use this tool ONLY for complex, multi-step, or ambiguous requests (e.g., "Plan a trip to Tokyo", "Compare iPhone 15 vs S24").

Break the request into 3–5 atomic, executable steps for worker agents. Each step must:
- Be a single action starting with a strong verb (e.g., Retrieve, Verify, Generate).
- Include all necessary context (who, what, where, when).
- Produce a concrete, verifiable result.
- Be self-contained and independent.

Return ONLY a valid JSON object with a "steps" array. Use the same language as the user query. Do not include explanations, markdown, or extra text.
"""

def get_plan_tool():
    async def plan_func(
        steps: Annotated[
            list[str],
            "List of plan steps. 针对当前任务拟定的计划或方案，将复杂问题分解为多个可执行步骤。",
        ] = "",
    ):
        logger.info(
            f"Generated plan: {steps}"
        )
        return json.dumps({"steps": steps})

    plan_tool = FunctionTool.from_defaults(
        async_fn=plan_func,
        name="planning-tool",
        description=PLAN_TOOL_DESCRIPTION,
        return_direct=False,
    )

    return plan_tool
