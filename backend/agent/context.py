from pydantic import BaseModel, Field
from typing import Dict, Any
from utils.time_utils import get_current_time_str


class RunContext(BaseModel):
    """Agent execution context variables.

    Holds runtime context information that can be injected into prompts.
    Automatically populates current_datetime if not provided.
    """
    current_datetime: str = Field(default_factory=get_current_time_str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] = None) -> "RunContext":
        """Construct RunContext from a dictionary.

        Args:
            data: Dictionary of context variables. If None, creates empty context.

        Returns:
            RunContext instance with current_datetime auto-filled if not provided.
        """
        if data is None:
            data = {}
        return cls(**data)

    def to_string(self) -> str:
        """Convert context to string format for prompt injection.

        Returns:
            Formatted string representation of all context variables.
        """
        context_items = []
        for key, value in self.model_dump().items():
            # Format key as human-readable (convert snake_case to Title Case)
            formatted_key = key.replace('_', ' ').title()
            context_items.append(f"- {formatted_key}: {value}")

        return "\n".join(context_items)
