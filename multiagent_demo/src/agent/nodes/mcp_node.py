# agent/mcp_node.py
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.react_executor import run_react_agent
from agent.config import AgentConfig
from agent.utils.helpers import extract_task_description

# MCP 工具缓存
_mcp_tools_cache = None


async def get_mcp_tools():
    """Get MCP tools with caching."""
    global _mcp_tools_cache
    
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    
    mcp_server_config = {
        "amap": {
            "url": AgentConfig.MCP_AMAP_URL,
            "headers": AgentConfig.MCP_AMAP_HEADERS,
            "transport": "sse"
        }
    }
    
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
        print(f"成功加载 {len(available_mcp_tools)} 个MCP工具: {[t.name for t in available_mcp_tools]}")
        _mcp_tools_cache = available_mcp_tools
        return available_mcp_tools
    except Exception as e:
        print(f"⚠️ 无法加载 MCP 工具: {e}")
        return []


def clear_mcp_tools_cache():
    """Clear MCP tools cache (useful for testing or reconfiguration)."""
    global _mcp_tools_cache
    _mcp_tools_cache = None

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
    
    task_desc = extract_task_description(state["messages"])
    
    final_answer = await run_react_agent(
        tools=mcp_tools,
        task_description=task_desc,
        language=AgentConfig.REACT_DEFAULT_LANGUAGE,
        system_prompt="你是一个地图导航专家，擅长处理地理位置、路线规划和POI搜索。"
    )

    print(f"Map Navigator finished: {final_answer[:100]}...")

    return {
        "messages": [AIMessage(content=final_answer, name="map_navigator")],
        "next": "supervisor"
    }