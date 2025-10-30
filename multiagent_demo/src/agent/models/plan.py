from typing_extensions import TypedDict
from enum import Enum
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field

class Status(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SubTask(BaseModel):
    name: str = Field(description="Short name of the subtask")
    description: str = Field(description="Detailed description of what to do")
    expected_outcome: str = Field(description="What the output should look like")
    outcome: Optional[str] = None
    assignee: str = Field(description="Who should execute this subtask: 'researcher', 'kb_retriever', 'map_navigator' or 'reporter'")
    state: Status = Status.TODO


class Plan(BaseModel):
    """The plan model used in the plan module, contains a list of subtasks."""
    
    id: str
    name: str
    description: str
    expected_outcome: str
    state: Status = Status.TODO
    subtasks: List[SubTask] = Field(default_factory=list)
    outcome: Optional[str] = None
    created_at: str | None = Field(
        description="The time the plan was created.",
        default=None,
        exclude=True,
    )
    # Result related fields
    finished_at: str | None = Field(
        description="The time the plan was finished.",
        default=None,
        exclude=True,
    )
