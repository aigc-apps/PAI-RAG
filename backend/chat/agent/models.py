from typing_extensions import TypedDict
from enum import Enum


class StepStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"


class PlanOutput(TypedDict):
    steps: list[str] = []
