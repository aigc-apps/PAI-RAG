"""Build the single ``Agent`` used by the SDK runner.

Reuses :func:`backend.agent_service.build_system_prompt` so the legacy and
SDK runners read the same prompt body + memory index + skills inventory.

The model is bound at Agent construction time. Callers pass either a model
name string (``runtime_config.get_active_model()`` or a per-run override) or
a fully-built ``Model`` instance — the runner now passes the latter via
:func:`backend.agents_sdk.runtime_setup.acquire_request_model` so each
request carries its own ``AsyncOpenAI`` (per-request key rotation against
the multi-key ``provider_pool``). On a mid-conversation model switch (via
``/v1/models/active``) the next call to ``build`` returns a fresh Agent and
the runner attaches it on ``RunState.from_string(initial_agent=...)``.
"""
from __future__ import annotations

from typing import Iterable

from agents import Agent, ModelSettings
from agents.tool import FunctionTool

from backend.agent_service import build_system_prompt
from backend.session_store_base import SERVER_USER_ID

AGENT_NAME = 'pai-rag'


def build(
    *,
    model: str,
    tools: Iterable[FunctionTool],
    user_id: str = SERVER_USER_ID,
    instructions_override: str | None = None,
) -> Agent:
    """Return an ``Agent`` configured with the given model + tools.

    ``instructions_override`` is for unit tests; production code lets
    :func:`build_system_prompt` assemble the prompt from disk so any memory
    index changes mid-session propagate at next ``build``.
    """
    instructions = instructions_override if instructions_override is not None else build_system_prompt(user_id=user_id)
    tool_list = list(tools)
    tool_names = {getattr(tool, 'name', '') for tool in tool_list}
    tool_use_behavior = (
        {'stop_at_tool_names': ['final_report']}
        if 'final_report' in tool_names
        else 'run_llm_again'
    )
    return Agent(
        name=AGENT_NAME,
        instructions=instructions,
        tools=tool_list,
        model=model,
        tool_use_behavior=tool_use_behavior,
        # Force ``stream_options.include_usage=true`` on chat-completions
        # streaming. The SDK's ``ChatCmplHelpers.get_stream_options_param``
        # only opts in by default for the official OpenAI client; against
        # qwen-plus / dashscope-compatible endpoints it leaves the flag
        # unset, the upstream omits ``usage`` from the stream, and our
        # ``response.completed`` would carry ``usage: null``. Setting it
        # here makes the request-level option explicit for every provider.
        model_settings=ModelSettings(include_usage=True),
    )
