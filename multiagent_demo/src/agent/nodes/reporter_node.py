# agent/reporter_node.py
from langchain_core.messages import HumanMessage, AIMessage
from agent.utils.llm_factory import get_llm

def extract_research_summary_for_reporter(state):
    """
    Extract user query and all researcher outputs from the state,
    and format them into a clean research summary for the reporter.
    """
    messages = state["messages"]
    
    user_query = ""
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.name is None:  # original user message
            user_query = msg.content
            break

    researcher_tasks = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            researcher_tasks.append({
                "task": messages[idx - 1].content if idx > 0 else "Unknown Task",
                "research_content": msg.content.strip()
            })
    
    sections = []
    
    for output in researcher_tasks:
        sections.append(f"【{output['task']}】\n{output['research_content']}")

    research_summary = "\n\n".join(sections)
    
    return user_query, research_summary

async def reporter_node(state):
    # Extract researcher's output
    user_query, research_summary = extract_research_summary_for_reporter(state)
    current_subtask = state.get("plan").subtasks[state.get("current_subtask_idx")]
    report_guidelines = f"""[Task] {current_subtask.name}
    [Requirements] {current_subtask.description}
    [Expected Outcome] {current_subtask.expected_outcome}
    """
    prompt = f"""
    Your task is to generate a report that meets the specified requirements, based on the user's original query, the available research findings, and the reporting guidelines.
    
    User’s Original Request:
    {user_query}

    Research Findings:
    {research_summary}
    
    Report Guidelines:
    {report_guidelines}
    """
    
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    print(f"Reporter output: {response.content}")
    
    return {
            "messages": [AIMessage(content=response.content, name="reporter")],
            "next": "supervisor"
        }