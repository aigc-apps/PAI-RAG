from __future__ import annotations
from typing import List, Optional, Tuple
from agent.context import AgentContext, RunVars
from agent.message import Message, ToolCall
from agent.tools.base import ToolBox
from app.schemas import ResponsesRequest
from app.store.base import Item

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _item_text(content: dict) -> str:
    if "text" in content:
        return content["text"] or ""
    # tolerate OpenAI-style message content arrays
    parts = content.get("content")
    if isinstance(parts, list):
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if isinstance(parts, str):
        return parts
    return ""


def items_to_messages(items: List[Item]) -> List[Message]:
    """Convert stored conversation items (history source of truth) into agent Messages.
    Reasoning items are skipped (not replayed to the model)."""
    msgs: List[Message] = []
    for it in items:
        if it.type == "message":
            msgs.append(
                Message(role=it.role or "user", content=_item_text(it.content))
            )
        elif it.type == "function_call":
            c = it.content
            msgs.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=c.get("call_id", ""),
                            name=c.get("name", ""),
                            arguments=c.get("arguments", "") or "",
                        )
                    ],
                )
            )
        elif it.type == "function_call_output":
            c = it.content
            msgs.append(
                Message(
                    role="tool",
                    tool_call_id=c.get("call_id", ""),
                    content=c.get("output", ""),
                )
            )
        # type == "reasoning": skip
    return msgs


def _input_to_turn(req_input) -> Message:
    if isinstance(req_input, str):
        return Message(role="user", content=req_input)
    # list of items: take the last item's text (string content or content array)
    text = ""
    for it in req_input:
        if isinstance(it, dict):
            text = _item_text(it)
    return Message(role="user", content=text)


async def build_context(
    request: ResponsesRequest, store
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store, assemble the AgentContext the agent runs.
    Returns (ctx, conversation_id). Raises ValueError on previous_response_id/conversation
    conflict (the route maps that to HTTP 400)."""
    history_items: List[Item] = []
    conversation_id = request.conversation
    if request.previous_response_id or request.conversation:
        history_items = await store.resolve_history(
            previous_response_id=request.previous_response_id,
            conversation=request.conversation,
        )
        if request.previous_response_id:
            resp = await store.get_response(request.previous_response_id)
            if resp is not None and resp.conversation_id:
                conversation_id = resp.conversation_id

    # Fresh turn with no prior conversation: open one so the run has a stable
    # conversation_id to persist items under (route relies on this for linking).
    if conversation_id is None:
        conversation_id = (await store.create_conversation()).id

    ctx = AgentContext(
        system_prompt=request.instructions or DEFAULT_SYSTEM_PROMPT,
        history=items_to_messages(history_items),
        current_turn=_input_to_turn(request.input),
        attachments=[],
        hints=[],
        tools=ToolBox([]),
        run_vars=RunVars(),
    )
    return ctx, conversation_id
