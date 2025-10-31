from typing import Literal
from langgraph.graph import StateGraph, START, END
import asyncio
from agent.nodes.supervisor_node import AgentState, _build_supervisor_subgraph
from agent.nodes.researcher_node import researcher_node
from agent.nodes.reporter_node import reporter_node
from agent.nodes.mcp_node import amap_mcp_node
from agent.nodes.kb_retriever_node import kb_retriever_node
def _build_base_graph_agent():
    supervisor_agent = _build_supervisor_subgraph()
    builder = StateGraph(AgentState)
    builder.add_edge(START, "supervisor")
    
    # 关键：将整个子图作为一个节点
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("researcher", researcher_node)
    builder.add_node("kb_retriever", kb_retriever_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("map_navigator", amap_mcp_node)
    builder.add_conditional_edges("supervisor", lambda state: state.get("next","researcher"), {
            "researcher": "researcher",
            "kb_retriever": "kb_retriever",
            "reporter": "reporter",
            "map_navigator": "map_navigator",
            "FINISH": END
        })
    
    # Workers always return to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("kb_retriever", "supervisor")
    builder.add_edge("reporter", "supervisor")
    builder.add_edge("map_navigator", "supervisor")
    
    return builder.compile()

graph = _build_base_graph_agent()

async def test_print():
    async for chunk in graph.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "写一个关于AI的两句话的博客文章"
                }
            ]
        }
    ):
        print(chunk)
        print("\n")
            
if __name__ == "__main__":
    asyncio.run(test_print())