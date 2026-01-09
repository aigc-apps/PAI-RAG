from pydantic import BaseModel
from datetime import datetime
import json
from llama_index.core.tools import FunctionTool
from loguru import logger
from functools import partial


class ThoughtRecord(BaseModel):
    timestamp: str
    thought: str


def record_thought(cache, cache_key: str, thought: str):
    timestamp = datetime.now().isoformat()
    record = ThoughtRecord(timestamp=timestamp, thought=thought)

    # 写入缓存
    if cache_key not in cache:
        cache[cache_key] = []
    cache[cache_key].append(record)

    logger.info(f"[{timestamp}] Thought recorded: {thought[:50]}...")
    logger.info(f"Total thoughts recorded: {len(cache[cache_key])} for key {cache_key}")
    return json.dumps(
        {
            "type": "text",
            "text": (
                f"Thought recorded: {thought[:50]}..." if len(thought) > 50 else thought
            ),
            "thoughts_count": len(cache[cache_key]),
        },
        ensure_ascii=False,
    )


def get_think_function(cache_key: str):
    return partial(record_thought, cache={}, cache_key=cache_key)


async def aget_simple_think_tool(cache_key: str):
    add_thought_func = get_think_function(cache_key)

    async def simple_think_handler(thought: str):
        return add_thought_func(thought=thought)

    think_tool = FunctionTool.from_defaults(
        async_fn=simple_think_handler,
        name="think",
        description="记录思考内容。用于复杂推理或缓存记忆。",
        return_direct=False,
    )
    openai_tools = []
    tools_name_to_fn = {}
    tool_name = "think"
    tools_name_to_fn[tool_name] = think_tool
    tool_metadata = think_tool.metadata
    tool_metadata.name = tool_name
    openai_tools.append(tool_metadata.to_openai_tool(skip_length_check=True))

    return openai_tools, tools_name_to_fn
