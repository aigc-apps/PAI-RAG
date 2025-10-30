from typing import List, Optional
import uuid
from typing import Annotated
from agent.models.plan import Plan, Status, SubTask


class PlanManager:
    def __init__(self, plan: Optional[Plan] = None):
        self.current_plan = plan

    async def create_plan(
        self,
        name: Annotated[str, "The plan name, should be concise, descriptive and not exceed 10 words"],
        description: Annotated[str, "The plan description, including the constraints, target and outcome to be achieved. The description should be clear, specific and concise, and all the constraints, target and outcome should be specific and measurable."],
        expected_outcome: Annotated[str, "The expected outcome of the plan, which should be specific, concrete and measurable."],
        subtasks: Annotated[
            Optional[List[dict]], 
            "A list of sequential sub-tasks that make up the plan. Each sub-task is represented as a dictionary with keys: 'name', 'description', and 'expected_outcome'. The sub-tasks should be clear, specific, and achievable, and should collectively lead to the successful completion of the overall plan."
        ] = None,
    ) -> str:
        """ Create a new plan. """
        plan_id = str(uuid.uuid4())
        parsed_subtasks = []
        if subtasks:
            for st in subtasks:
                parsed_subtasks.append(
                    SubTask(
                        name=st.get("name", ""),
                        description=st.get("description", ""),
                        expected_outcome=st.get("expected_outcome", ""),
                        assignee=st.get("assignee", "researcher"),
                    )
                )
        
        self.current_plan = Plan(
            id=plan_id,
            name=name,
            description=description,
            expected_outcome=expected_outcome,
            subtasks=parsed_subtasks,
        )
        return f"Plan '{name}' created successfully."

    async def update_subtask_state(
        self,
        subtask_idx: Annotated[int, "The index of the subtask to update"],
        state: Annotated[
            str, 
            "The new state of the subtask. MUST in [todo, in_progress, completed, failed]. If you want to mark a subtask as done, you SHOULD call `finish_subtask` instead with the specific outcome."
        ]
    ) -> str:
        """
        Update the state of a subtask by given index and state. Note if you want to mark a subtask as done, you SHOULD call `finish_subtask` instead with the specific outcome.
        """
        if self.current_plan is None:
            return "No plan exists. Please create a plan first."
        
        try:
            new_state = Status(state)
        except ValueError:
            return f"Invalid status: {state}. Must be one of: {[s.value for s in Status]}"

        try:
            subtask = self.current_plan.subtasks[subtask_idx]
        except IndexError:
            return f"Subtask at index {subtask_idx} not found." 
        subtask.state = new_state
        return f"Subtask '{subtask.name}' status updated to {new_state}."

    async def finish_subtask(
        self,
        subtask_idx: Annotated[int, "The index of the subtask to update"],
        subtask_outcome: Annotated[str, "The outcome of the subtask"],
    ) -> str:
        """Label the subtask as done by given index and outcome."""
        
        if self.current_plan is None:
            return "No plan exists."
        try:
            subtask = self.current_plan.subtasks[subtask_idx]
        except IndexError:
            return f"Subtask at index {subtask_idx} not found."
        subtask.state = Status.COMPLETED
        subtask.outcome = subtask_outcome
        return f"Subtask '{subtask.name}' finished. Outcome recorded."  

    async def finish_plan(
        self,
        outcome: Annotated[str, "The final report of the plan, summarizing the whole process and outcome."],
    ) -> str:
        """Finish the current plan and provide a final report."""
        if self.current_plan is None:
            return "No plan exists."
        self.current_plan.state = Status.COMPLETED
        self.current_plan.outcome = outcome
        plan_name = self.current_plan.name
        # self.current_plan = None  # Clear the current plan
        return f"Plan '{plan_name}' finished. Final report recorded."
    

