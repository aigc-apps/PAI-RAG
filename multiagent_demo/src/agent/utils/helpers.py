"""Helper utilities for agent nodes."""
from langchain_core.messages import SystemMessage
from typing import List, Any


def extract_task_description(messages: List[Any]) -> str:
    """Extract task description from message list.
    
    Args:
        messages: List of message objects from state.
        
    Returns:
        Task description string, or "No task provided." if not found.
    """
    for msg in reversed(messages):
        if isinstance(msg, SystemMessage):
            return msg.content
    return "No task provided."
