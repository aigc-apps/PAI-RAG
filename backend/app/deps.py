from __future__ import annotations
from dataclasses import dataclass
from fastapi import Request
from agent.agent import Agent
from agent.budgeting import AgentMessageManager


@dataclass
class AppState:
    store: object  # ResponseStore
    llm: object  # has .astream(messages, tools)
    default_model: str
    context_window: int = 110000
    max_output_tokens: int = 8000

    def make_agent(self) -> Agent:
        # Verified signature: Agent(llm, max_steps=..., budget: Optional[AgentMessageManager]=None).
        # Pass an explicit budget from AppState's window so fake/echo test LLMs (which
        # lack a `context_window` attr) don't get a zero-width budget that truncates input.
        return Agent(
            llm=self.llm,
            budget=AgentMessageManager(
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
