from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from agent.models.plan_manager import PlanManager
from agent.models.plan import Plan, SubTask, Status
from langgraph.graph import MessagesState
from pydantic import Field, BaseModel
from typing import List
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from datetime import datetime

ENABLED_WORKERS = ["researcher", "kb_retriever", "reporter", "map_navigator"]

class SupervisorState(MessagesState):
    plan: Optional[Plan] = None
    next: str = "supervisor"
    current_subtask_idx: Optional[int] = None
    supervisor_step: Literal[
        "reason_about_plan",
        "create_plan",
        "assign_next_subtask",
        "check_completion",
        "done"
    ] = "reason_about_plan"
    

class CreatePlanToolSchema(BaseModel):
    """Create a plan with subtasks to achieve the user's goal."""
    name: str = Field(description="Name of the overall plan")
    description: str = Field(description="High-level description of the plan")
    expected_outcome: str = Field(description="Final expected result of the whole plan")
    subtasks: List[SubTask] = Field(description="List of subtasks to execute")
    

create_plan_tool = StructuredTool.from_function(
    func=lambda **kwargs: kwargs,  # dummy function
    name="create_plan",
    description="Create a plan with subtasks, answer with the same language as the user’s query.",
    args_schema=CreatePlanToolSchema,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ===== 1. 自动完成上一个子任务 =====
async def finalize_subtask(state: SupervisorState):
    current_idx = state.get("current_subtask_idx")
    messages = state["messages"]
    plan = state.get("plan")

    # 如果没有正在处理的子任务，直接进入计划推理
    if current_idx is None or current_idx < 0:
        return {"supervisor_step": "reason_about_plan"}

    # 检查最后一条消息是否来自 worker
    if not (messages and isinstance(messages[-1], AIMessage) and messages[-1].name in ENABLED_WORKERS):
        # 没有有效回复,可能异常，保守进入推理
        return {"supervisor_step": "reason_about_plan"}

    worker_output = messages[-1].content
    pm = PlanManager(plan=plan)

    # 标记为完成
    await pm.finish_subtask(current_idx, worker_output)
    return {
        "plan": pm.current_plan,
        "current_subtask_idx": None,
        "supervisor_step": "reason_about_plan"
    }

# ===== 2. Reason: 是否需要计划 =====
async def reason_about_plan(state: SupervisorState):
    plan = state.get("plan")
    pm = PlanManager(plan=plan)

    if pm.current_plan is None or len(pm.current_plan.subtasks) == 0:
        next_step = "create_plan"
    else:
        next_step = "check_completion"

    result = {"supervisor_step": next_step, "plan": pm.current_plan}

    return result

# ===== 3. Act: 创建计划（使用 tool_call）=====
async def create_plan(state: SupervisorState):
    messages = state["messages"]
    goal = messages[0].content if messages else "No goal"
    
    planning_messages = [
        SystemMessage(content=(
            """You are a planning supervisor responsible for orchestrating a team to answer complex questions that may require both internal knowledge and external research. Use the 'create_plan' tool to generate a detailed, step-by-step plan.

Each subtask MUST be assigned to exactly one of the following roles using the 'assignee' field:
- 'kb_retriever': Retrieve relevant information from the company's private knowledge base and utilize it for matters involving the proprietary knowledge base data of Starry Technology Co., Ltd., which includes the company's remote work policies, employee survey data, IT system log summaries, and more.
- 'researcher': Conduct web searches to gather up-to-date external information (e.g., industry trends, academic studies, public regulations, competitor insights). Do NOT use this for internal company matters.
- 'map_navigator': Provide geographic, location-based, or routing information only when the query involves physical places, travel, or spatial context.
- 'reporter': Synthesize inputs from 'kb_retriever', 'researcher' and 'map_navigator' to produce the final output (e.g., analysis report, recommendation memo, summary). This role does not perform retrieval.


⚠️ Rules:
1. NEVER assign tasks to any role outside the four listed above.
2. For questions involving the company’s remote work policies, employee survey data or IT system log summaries, assign a subtask to 'kb_retriever'.
3. For general or public-domain knowledge, assign to 'researcher'.
4. Respond in the same language as the user’s query.

Ensure the plan is logical, minimally redundant, and leverages internal and external sources appropriately.
"""
        )),
        HumanMessage(content=f"User goal: {goal}, please create a plan to achieve this goal using the same language as the user’s query.")
    ]
    
    llm_with_tools = llm.bind_tools([create_plan_tool])
    response: AIMessage = await llm_with_tools.ainvoke(planning_messages)
    
    plan = state.get("plan")
    pm = PlanManager(plan=plan)
    
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "create_plan":
                try:
                    await pm.create_plan(**tc["args"])
                    return {"plan": pm.current_plan, "supervisor_step": "check_completion"}
                except Exception as e:
                    print(f"Plan creation failed: {e}")
    
    # Fallback
    await pm.create_plan(
        name="Default Plan",
        description="Fallback execution",
        expected_outcome="Task done",
        subtasks=[
            SubTask(
                name="Research",
                description="Gather info",
                expected_outcome="Notes",
                assignee="researcher"
            ),
            SubTask(
                name="Report",
                description="Write output",
                expected_outcome="Final text",
                assignee="reporter"
            )
        ]
    )
    return {"plan": pm.current_plan, "supervisor_step": "check_completion"}

# ===== 4. Check: 是否全部完成？ =====
async def check_completion(state: SupervisorState):
    pm = PlanManager(plan=state.get("plan"))
    
    if pm.current_plan and all(st.state == Status.COMPLETED for st in pm.current_plan.subtasks):
        await pm.finish_plan("All done.")
        return {
            "plan": pm.current_plan,
            "next": "FINISH",
            "supervisor_step": "done"
        }
    else:
        return {"supervisor_step": "assign_next_subtask"}

# ===== 5. Assign: 分配下一个子任务 =====
async def assign_next_subtask(state: SupervisorState):
    pm = PlanManager(plan=state.get("plan"))
    if not pm.current_plan:
        return {"supervisor_step": "reason_about_plan"}
    
    # 找下一个 TODO
    next_idx = None
    for i, st in enumerate(pm.current_plan.subtasks):
        if st.state == Status.TODO:
            next_idx = i
            break
    
    if next_idx is None:
        return {"supervisor_step": "check_completion"}
    
    subtask = pm.current_plan.subtasks[next_idx]
    next_worker = subtask.assignee

    await pm.update_subtask_state(next_idx, Status.IN_PROGRESS)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    instruction = SystemMessage(
        content=f"- 今天的日期是：**{current_datetime}**\n"
                f"你的任务是：{subtask.description}。预期输出是：{subtask.expected_outcome}。"
    )
    
    return {
        "messages": state["messages"] + [instruction],
        "plan": pm.current_plan,
        "next": next_worker,
        "current_subtask_idx": next_idx,
        "supervisor_step": "done",
    }

def _build_supervisor_subgraph():
    builder = StateGraph(SupervisorState)
    
    builder.add_node("finalize_subtask", finalize_subtask)
    builder.add_node("reason_about_plan", reason_about_plan)
    builder.add_node("create_plan", create_plan)
    builder.add_node("check_completion", check_completion)
    builder.add_node("assign_next_subtask", assign_next_subtask)
    
    builder.set_entry_point("finalize_subtask")

    builder.add_edge("finalize_subtask", "reason_about_plan")
    
    def route_after_reason(state: SupervisorState):
        return state["supervisor_step"]
    
    builder.add_conditional_edges(
        "reason_about_plan",
        route_after_reason,
        {
            "create_plan": "create_plan",
            "check_completion": "check_completion"
        }
    )
    
    builder.add_edge("create_plan", "check_completion")
    
    
    builder.add_conditional_edges(
        "check_completion",
        lambda s: s["supervisor_step"],
        {
            "assign_next_subtask": "assign_next_subtask",
            "done": END
        }
    )
    
    # assign → END（交出控制权给 worker）
    builder.add_edge("assign_next_subtask", END)
    
    return builder.compile()