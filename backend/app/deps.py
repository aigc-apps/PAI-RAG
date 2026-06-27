from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from fastapi import Request
from agent.agent import Agent
from agent.budgeting import AgentMessageManager
from agent.soul import Soul, DEFAULT_SOUL
from agent.tools.registry import ToolRegistry
from app.runs import RunManager
from app.providers import ProviderRouter


@dataclass
class AppState:
    store: object  # ResponseStore
    llm: object  # has .astream(messages, tools)
    default_model: str
    context_window: int = 110000
    max_output_tokens: int = 8000
    soul: Soul = field(default_factory=lambda: DEFAULT_SOUL)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    runs: RunManager = field(default_factory=RunManager)
    router: Optional[ProviderRouter] = None
    memory_enabled: bool = False
    memory_model: str = ""

    def make_agent(self, llm=None, context_window: Optional[int] = None,
                   max_output_tokens: Optional[int] = None) -> Agent:
        # Verified signature: Agent(llm, max_steps=..., budget: Optional[AgentMessageManager]=None).
        # Pass an explicit budget from AppState's window so fake/echo test LLMs (which
        # lack a `context_window` attr) don't get a zero-width budget that truncates input.
        return Agent(
            llm=llm if llm is not None else self.llm,
            budget=AgentMessageManager(
                context_window=context_window if context_window is not None else self.context_window,
                max_output_tokens=max_output_tokens if max_output_tokens is not None else self.max_output_tokens,
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
