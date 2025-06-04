from typing import List
from pydantic import BaseModel, Field
from datetime import datetime
import json
from llama_index.core.tools import FunctionTool


# 定义思考记录接口
class ThoughtRecord(BaseModel):
    timestamp: str
    thought: str


class ThinkParams(BaseModel):
    thought: str = Field(..., description="需要记录的思考内容")


# 全局日志（适用于单用户场景）
thoughts_log: List[ThoughtRecord] = []


def clear_thoughts_log():
    global thoughts_log
    thoughts_log = []


async def simple_think_handler(thought: str):
    params_model = ThinkParams(thought=thought)
    timestamp = datetime.utcnow().isoformat()
    thoughts_log.append(
        ThoughtRecord(timestamp=timestamp, thought=params_model.thought)
    )
    print(f"[{timestamp}] Thought recorded: {params_model.thought[:50]}...")
    return json.dumps(
        {
            "type": "text",
            "text": (
                f"Thought recorded: {params_model.thought[:50]}..."
                if len(params_model.thought) > 50
                else params_model.thought
            ),
            "thoughts_count": len(thoughts_log),  # 返回当前思考记录数量
        },
        ensure_ascii=False,
    )


async def aget_simple_think_tool():
    think_tool = FunctionTool.from_defaults(
        async_fn=simple_think_handler,
        name="think",
        description="记录思考内容。用于复杂推理或缓存记忆。",
    )
    openai_tools = []
    tools_name_to_fn = {}
    tool_name = "think"
    tools_name_to_fn[tool_name] = think_tool
    tool_metadata = think_tool.metadata
    tool_metadata.name = tool_name
    openai_tools.append(tool_metadata.to_openai_tool())

    return openai_tools, tools_name_to_fn
