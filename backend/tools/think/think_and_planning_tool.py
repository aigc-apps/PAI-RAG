from pydantic import BaseModel
import json
from llama_index.core.tools import FunctionTool
from loguru import logger
from functools import partial
from typing import Annotated, Optional, List


class ThoughtRecord(BaseModel):
    thought: str
    thought_number: int
    action: str
    plan: List[str]


def record_thought(
    think_cache: List[ThoughtRecord] = [],
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
    think_cache.append(thought_record)
    logger.info(f"Total thoughts recorded: {len(think_cache)}.")

    return json.dumps(
        {
            "status": "success",
            "message": "Thought recorded successfully.",
        },
        ensure_ascii=False,
    )


def get_think_function(think_cache: List[ThoughtRecord] = []):
    return partial(record_thought, think_cache=think_cache)


async def aget_simple_think_tool(think_cache: List[ThoughtRecord] = []):
    add_thought_func = get_think_function(think_cache)

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
            "基于当前思考和规划，建议下一步采取的行动步骤，可以是下一步需要调用的一个或多个工具的名称及其需要的参数，格式必须为字符串类型，不要输出JSON。你需要尽可能的调用多个并行的工具。",
        ] = "",
        plan: Annotated[
            list[str],
            "List of plan steps. 针对当前任务拟定的计划或方案，将复杂问题分解为多个可执行步骤。你需要尽可能的分解为几个并行的可执行步骤。",
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
        name="think-and-planning",
        description="这是用于系统化思考与规划的工具，支持用户在面对复杂问题或任务时，分阶段梳理思考、规划和行动步骤。工具强调思考（thought）、计划（plan）与实际行动（action）的结合，通过编号（thoughtNumber）追踪过程。该工具不会获取新信息或更改数据库，只会将想法附加到记忆中。当需要复杂推理或某种缓存记忆时，可以使用它。",
        return_direct=False,
    )

    return think_tool
