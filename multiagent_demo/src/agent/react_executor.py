# agent/utils/react_executor.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
)
from langchain_core.tools import BaseTool
from typing import List, Sequence, Optional, Any
from datetime import datetime

async def run_react_agent(
    tools: Sequence[BaseTool],
    task_description: str,
    system_prompt: Optional[str] = None,
    max_steps: int = 3,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    language: str = "中文"
) -> str:
    """
    通用 ReAct Agent 执行器，支持任意 LangChain 工具。
    
    Args:
        tools: 工具列表（如 [tavily_search, amap_poi_search]）
        task_description: 用户任务描述
        system_prompt: 可选系统提示（若未提供，使用默认）
        max_steps: 最大推理步数
        model: LLM 模型名称
        temperature: LLM 温度
        language: 回答语言（用于提示词）
        
    Returns:
        str: 最终回答
    """
    # 初始化 LLM 并绑定工具
    llm = ChatOpenAI(model=model, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools)

    # 构建系统提示
    if system_prompt is None:
        system_prompt = (
            f"你是一个智能助理，负责完成指定任务。请使用{language}进行思考和回答。"
        )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"- 当前时间：{current_time}\n- 你的任务是：{task_description}"
        )
    ]

    # ReAct 循环
    for step in range(max_steps):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 若无工具调用，返回最终答案
        if not response.tool_calls:
            return response.content or "未生成有效回答。"

        # 执行所有工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # 查找对应工具
            found_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    found_tool = tool
                    break

            if found_tool is None:
                tool_result = f"错误：未找到工具 '{tool_name}'"
            else:
                try:
                    # 执行工具（LangChain 工具接受 dict 或 str）
                    tool_result = await found_tool.ainvoke(tool_args)
                except Exception as e:
                    tool_result = f"工具 '{tool_name}' 执行失败: {str(e)}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            )

    # 超出步数，返回最后内容
    return messages[-1].content if messages[-1].content else "任务执行超时。"