"""``read_handle`` — recover a tool result that was offloaded from the model window.

The tiered-context compressor replaces an old, large tool result with a compact
placeholder carrying a ``store://tool/<call_id>`` handle (see
``agent.context_offload``). This tool resolves that handle back to the full,
verbatim body so the model can re-read the parts it now needs:

- in-run map first: a result offloaded *this run* is not persisted yet, so the full
  body is held in the ToolScope's per-run ``run_bodies`` map keyed by ``call_id``;
- then the durable store: for results from earlier runs, the ``resolver`` (the
  ResponseStore) reads the full ``function_call_output`` output, scoped to the
  caller's conversation via the ambient ToolScope — a client can never read another
  conversation's results because ``conversation_id`` comes from the scope, not args.

An optional ``start``/``count`` line range keeps a targeted re-read cheap instead of
re-inflating a huge body into the window.
"""
from __future__ import annotations

from typing import Optional

from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope
from agent.context_offload import parse_handle

_PARAMS = {
    "type": "object",
    "properties": {
        "handle": {
            "type": "string",
            "description": (
                "The handle from an offloaded tool-result placeholder, e.g. "
                "'store://tool/call_abc123' (a bare call id is also accepted)."
            ),
        },
        "start": {
            "type": "integer",
            "description": "Optional 0-based first line to return (default 0).",
        },
        "count": {
            "type": "integer",
            "description": "Optional number of lines to return from `start` (0 = all).",
        },
    },
    "required": ["handle"],
}

_DESCRIPTION = (
    "Fetch the full, original content of a tool result that was offloaded from the "
    "conversation to save context (shown as an '[offloaded tool result]' placeholder "
    "with a handle). Use it to re-read details you no longer see inline. Optionally "
    "pass start/count to read only a line range."
)


def make_read_handle_tool(resolver=None) -> Tool:
    """Build ``read_handle`` bound to ``resolver`` (a ResponseStore exposing
    ``get_tool_result(conversation_id, call_id)``). ``resolver`` may be None in
    contexts without a store — the in-run cache still works for same-run recovery."""

    async def fn(handle: str = "", start: int = 0, count: int = 0, **_ignored) -> str:
        call_id = parse_handle(handle)
        if not call_id:
            return "[read_handle] a valid handle is required."

        scope = get_current_tool_scope()
        # In-run bodies first (offloaded but not yet persisted this run)...
        run_bodies = getattr(scope, "run_bodies", None) or {}
        body: Optional[str] = run_bodies.get(call_id)
        # ...then the durable store, scoped to the caller's conversation.
        if body is None and resolver is not None:
            cid = getattr(scope, "conversation_id", None)
            if cid:
                try:
                    body = await resolver.get_tool_result(cid, call_id)
                except Exception as exc:  # noqa: BLE001 — surface as text, never raise
                    return f"[read_handle] lookup failed: {exc}"

        if body is None:
            return (
                f"[read_handle] no stored content for handle '{handle}'. It may have "
                "been produced in a different conversation, or never offloaded."
            )

        if count and count > 0:
            lines = body.splitlines()
            begin = max(0, int(start))
            body = "\n".join(lines[begin:begin + int(count)])
        return body

    return Tool(name="read_handle", description=_DESCRIPTION, parameters=_PARAMS, fn=fn)
