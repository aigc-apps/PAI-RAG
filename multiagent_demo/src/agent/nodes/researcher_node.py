# agent/researcher_node.py
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from agent.react_executor import run_react_agent
from langchain.agents import create_agent
tavily_search_tool = TavilySearch(max_results=5)

# =========== 方式一：直接复用Langgraph封装的react agent ===========

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# agent = create_agent(llm, [tavily_search_tool])

# async def researcher_node(state):
#     # Find last SystemMessage for task
#     task_desc = "No task provided."
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, SystemMessage):
#             task_desc = msg.content
#             break
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     agent_input = {
#         "messages": [HumanMessage(content=f"- 今天的日期是：**{current_datetime}** \n 你的任务是: {task_desc}， 请使用中文进行搜索")]
#     }
#     result = await agent.ainvoke(agent_input, config={"recursion_limit": 10})
#     final_content = result["messages"][-1].content

#     print(f"Researcher finished: {final_content[:100]}...")
#     return {
#         "messages": [AIMessage(content=final_content, name="researcher")],
#         "next": "supervisor"
#     }

# =========== 方式二：基于langchain自己实现 react agent ===========

async def researcher_node(state):
    # 提取任务
    task_desc = "No task provided."
    for msg in reversed(state["messages"]):
        if isinstance(msg, SystemMessage):
            task_desc = msg.content
            break

    # 调用通用 ReAct 执行器
    final_answer = await run_react_agent(
        tools=[tavily_search_tool],
        task_description=task_desc,
        language="中文"
    )

    print(f"Researcher finished: {final_answer[:100]}...")

    return {
        "messages": [AIMessage(content=final_answer, name="researcher")],
        "next": "supervisor"
    }


# =========== 流式输出：直接复用Langgraph封装的react agent stream ===========

# from langgraph.graph import StateGraph, START
# from agent.nodes.supervisor_node import SupervisorState
# import asyncio

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# agent = create_agent(llm, [tavily_search_tool])

# async def stream_researcher_node(state):
#     task_desc = "No task provided."
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, SystemMessage):
#             task_desc = msg.content
#             break
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     agent_input = {
#         "messages": [HumanMessage(content=f"- 今天的日期是：**{current_datetime}** \n 你的任务是: {task_desc}， 请使用中文进行搜索")]
#     }
#     full_content = ""
#     async for token, metadata in agent.astream(agent_input, config={"recursion_limit": 10}, stream_mode="messages"):
#         # 流式打印 LLM tokens
#         print(f"node: {metadata['langgraph_node']}")
#         print(f"content: {token.content_blocks}")
#         print("\n")
#         if len(token.content_blocks) > 0 and token.content_blocks[0]['type'] == 'text':
#             full_content += token.content_blocks[0]['text']
#         yield {
#             "messages": [AIMessage(content=full_content, name="researcher")],
#             "next": "supervisor"
#         }

# async def test_print():
#     """
#     测试流式输出
#     """
#     graph = (
#         StateGraph(SupervisorState)
#         .add_node("researcher_node", stream_researcher_node)
#         .add_edge(START, "researcher_node")
#         .compile()
#     )
    
#     async for event in graph.astream(
#         {"messages": [SystemMessage(content="2025年10月29日阿里巴巴股票价格是多少")]},
#         stream_mode="messages"
#     ):
#         print(event, end="", flush=True)
    
#     print("\n\n=== 流式输出完成 ===")

# if __name__ == "__main__":
#     asyncio.run(test_print())