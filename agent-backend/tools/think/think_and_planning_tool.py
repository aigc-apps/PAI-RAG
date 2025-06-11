from pydantic import BaseModel
import json
from llama_index.core.tools import FunctionTool
from loguru import logger
from functools import partial
from typing import Annotated, Optional, List, Dict


class ThoughtRecord(BaseModel):
    thought: str
    thought_number: int
    action: str
    plan: List[str]


def _format_plan(plan: Dict) -> str:
    """Format a plan for display."""
    output = f"思考: {plan['thought']} \n\n"
    output += "计划: \n"
    for i, step in enumerate(plan["plan"]):
        output += f"{i}. {step}\n"
    output += "\n\n"
    output += f"下一步行动: {plan['action']} \n\n"

    return output


def record_thought(
    cache,
    cache_key: str,
    thought: Optional[str] = None,
    thought_number: Optional[int] = None,
    action: Optional[str] = None,
    plan: Optional[List[str]] = None,
):

    thought_record = ThoughtRecord(
        thought=thought or "",
        thought_number=thought_number or 1,
        action=action or "",
        plan=plan or [],
    )
    # 写入缓存
    if cache_key not in cache:
        cache[cache_key] = []
    cache[cache_key].append(thought_record)
    logger.info(
        f"Recorded thought: {thought_record} for key {cache_key}. Total thoughts recorded: {len(cache[cache_key])}."
    )

    return json.dumps(
        {
            "thought": thought,
            "thought_number": thought_number,
            "action": action,
            "plan": "\n".join(plan) if plan else "",
            "thoughts_count": len(cache[cache_key]),
        },
        ensure_ascii=False,
    )


def get_think_function(cache_key: str):
    return partial(record_thought, cache={}, cache_key=cache_key)


async def aget_simple_think_tool(cache_key: str):
    add_thought_func = get_think_function(cache_key)

    async def simple_think_handler(
        thought: Annotated[
            str,
            "当前的思考内容，可以是对问题的分析、假设、洞见、反思或对前一步骤的总结。强调深度思考和逻辑推演，是每一步的核心。",
        ] = "",
        thought_number: Annotated[
            int,
            "当前思考步骤的编号，用于追踪和回溯整个思考与规划过程，便于后续复盘与优化。如果是第一次思考，则编号为1。后续思考步骤的编号应递增，表示思考的进展和深化。",
        ] = 1,
        action: Annotated[
            str,
            "基于当前思考和规划，建议下一步采取的行动步骤，要求具体、可执行、可验证，可以是下一步需要调用的一个或多个工具。",
        ] = "",
        plan: Annotated[
            list[str],
            "List of plan steps. 针对当前任务拟定的计划或方案，将复杂问题分解为多个可执行步骤。",
        ] = "",
    ):
        logger.info(
            f"Thinking: {thought}, Thought Number: {thought_number}, Action: {action}, Plan: {plan}"
        )
        return add_thought_func(
            thought=thought, thought_number=thought_number, action=action, plan=plan
        )

    think_tool = FunctionTool.from_defaults(
        async_fn=simple_think_handler,
        name="think_and_planning",
        description="这是用于系统化思考与规划的工具，支持用户在面对复杂问题或任务时，分阶段梳理思考、规划和行动步骤。工具强调思考（thought）、计划（plan）与实际行动（action）的结合，通过编号（thoughtNumber）追踪过程。该工具不会获取新信息或更改数据库，只会将想法附加到记忆中。当需要复杂推理或某种缓存记忆时，可以使用它。",
    )
    openai_tools = []
    tools_name_to_fn = {}
    tool_name = "think_and_planning"
    tools_name_to_fn[tool_name] = think_tool
    tool_metadata = think_tool.metadata
    tool_metadata.name = tool_name
    openai_tools.append(tool_metadata.to_openai_tool())

    return openai_tools, tools_name_to_fn
