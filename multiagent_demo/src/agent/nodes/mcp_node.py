# agent/mcp_node.py
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.react_executor import run_react_agent
from langchain.agents import create_agent

async def get_mcp_tools():
    mcp_server_config = {
        "amap": {
            "url": "https://mcp-server-amap-jitptfyoyw.cn-hangzhou.fcapp.run/sse",
            "headers": "",
            "transport": "sse"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
        print(f"成功加载 {len(available_mcp_tools)} 个MCP工具: {[t.name for t in available_mcp_tools]}")
        return available_mcp_tools
    except Exception as e:
        print(f"⚠️ 无法加载 MCP 工具: {e}")
        return []

# =========== 方式一：直接复用Langgraph封装的react agent ===========

# async def amap_mcp_node(state):
#     # Find last SystemMessage for task
#     llm = ChatOpenAI(model="gpt-4o", temperature=0)
#     agent = create_agent(llm, await get_mcp_tools())
#     task_desc = "No task provided."
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, SystemMessage):
#             task_desc = msg.content
#             break
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     agent_input = {
#         "messages": [HumanMessage(content=f"- 今天的日期是：**{current_datetime}** \n 你的任务是: {task_desc}")]
#     }
#     result = await agent.ainvoke(agent_input, config={"recursion_limit": 10})
#     final_content = result["messages"][-1].content

#     print(f"Researcher finished: {final_content[:100]}...")
#     return {
#         "messages": [AIMessage(content=final_content, name="map_navigator")],
#         "next": "supervisor"
#     }

# =========== 方式二：基于langchain自己实现 react agent ===========

async def amap_mcp_node(state):
    mcp_tools = await get_mcp_tools()
    
    task_desc = "No task provided."
    for msg in reversed(state["messages"]):
        if isinstance(msg, SystemMessage):
            task_desc = msg.content
            break
    
    final_answer = await run_react_agent(
        tools=mcp_tools,
        task_description=task_desc,
        language="中文",
        system_prompt="你是一个地图导航专家，擅长处理地理位置、路线规划和POI搜索。"
    )

    print(f"Map Navigator finished: {final_answer[:100]}...")

    return {
        "messages": [AIMessage(content=final_answer, name="map_navigator")],
        "next": "supervisor"
    }