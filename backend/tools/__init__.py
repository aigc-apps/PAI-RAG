"""Agents-SDK ``@function_tool`` wrappers around the existing
:class:`tools.GenericHandler` methods.

Each wrapper:

1. Reads ``ctx.context.handler`` (a per-run ``GenericHandler`` instance
   prepared by :mod:`backend.agents_sdk.runner`).
2. Calls the matching legacy ``do_*`` method.
3. Returns the ``StepOutcome.data`` payload to the SDK as the tool result.

``ask_user`` and any future approval-gated tools carry
``needs_approval=True`` so the SDK surfaces them as
``ToolApprovalItem`` interruptions. The runner either maps them to the two
OpenAI wires when HITL is explicitly enabled, or resolves them autonomously
with a conservative default.
"""
from __future__ import annotations

from backend.tools.wrappers import TOOLS, build_tool_list

__all__ = ['TOOLS', 'build_tool_list']
