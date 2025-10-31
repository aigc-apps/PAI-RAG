# agent/kb_retriever_node.py
from langchain_core.messages import AIMessage
from agent.tools.knowledge_base_tool import knowledge_base_search  # 导入自定义知识库检索工具
from agent.react_executor import run_react_agent
from agent.config import AgentConfig
from agent.utils.helpers import extract_task_description

# =========== 方式一：直接复用Langgraph封装的react agent ===========

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# # 创建 agent，绑定自定义知识库工具
# kb_agent = create_agent(llm, [knowledge_base_search])

# async def kb_retriever_node(state):
#     # 提取任务描述（最后一个 SystemMessage）
#     task_desc = "No task provided."
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, SystemMessage):
#             task_desc = msg.content
#             break

#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     agent_input = {
#         "messages": [
#             HumanMessage(
#                 content=f"- 今天的日期是：**{current_datetime}**\n"
#                         f"你的任务是：{task_desc}。请使用中文，并优先调用知识库搜索工具获取信息。"
#             )
#         ]
#     }

#     result = await kb_agent.ainvoke(agent_input, config={"recursion_limit": 10})
#     final_content = result["messages"][-1].content

#     print(f"KB Researcher finished: {final_content[:100]}...")
#     return {
#         "messages": [AIMessage(content=final_content, name="kb_retriever")],
#         "next": "supervisor"
#     }


# =========== 方式二：基于langchain自己实现 react agent ===========

async def kb_retriever_node(state):
    task_desc = extract_task_description(state["messages"])

    final_answer = await run_react_agent(
        tools=[knowledge_base_search],
        task_description=task_desc,
        language=AgentConfig.REACT_DEFAULT_LANGUAGE,
        system_prompt="你只能访问公司内部知识库，不得进行外部搜索或猜测。"
    )

    return {
        "messages": [AIMessage(content=final_answer, name="kb_retriever")],
        "next": "supervisor"
    }