from __future__ import annotations
from typing import List, Optional, Tuple

MEMORY_INJECT_LIMIT = 30
from agent.context import AgentContext, RunVars
from agent.message import Message, ToolCall
from agent.tools.base import ToolBox
from agent.tools.registry import ToolRegistry
from app.schemas import ResponsesRequest
from app.store.base import Item, new_conversation_id
from agent.soul import Soul, DEFAULT_SOUL, render_stable_system_prompt, render_context_block


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
    # list of items: prefer the LAST item whose role is "user" or absent
    text = ""
    for it in req_input:
        if isinstance(it, dict) and it.get("role", "user") in ("user", None):
            t = _item_text(it)
            if t:
                text = t
    return Message(role="user", content=text)


async def build_context(
    request: ResponsesRequest,
    store,
    *,
    soul: Soul = DEFAULT_SOUL,
    registry: Optional[ToolRegistry] = None,
    project_context: str = "",
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store and assemble the AgentContext.
    Composes the effective soul (default <- request.soul <- instructions) into a
    layered system prompt. Selects tools from the registry governed by
    soul.tools_enabled; registry=None keeps empty-ToolBox behavior.
    Raises ValueError on previous_response_id/conversation conflict (-> HTTP 400)."""
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

    if conversation_id is None:
        conversation_id = new_conversation_id()

    override = dict(request.soul or {})
    effective_soul = soul.merge(override)

    if registry is not None:
        selected = (
            effective_soul.tools_enabled
            if effective_soul.tools_enabled is not None
            else registry.names()
        )
        toolbox = registry.build_toolbox(selected)
    else:
        toolbox = ToolBox([])

    tool_names = [t.name for t in toolbox.tools]

    system_prompt = render_stable_system_prompt(
        effective_soul, tool_names=tool_names, project_context=project_context
    )

    summary = ""
    if conversation_id:
        conv = await store.get_conversation(conversation_id)
        if conv is not None and conv.summary:
            summary = conv.summary
            history_items = [it for it in history_items if it.seq > conv.summarized_seq]

    memories: List[str] = []
    uid = request.resolved_user_id
    if uid:
        memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)]
    instructions = "\n\n".join(s for s in [
        (effective_soul.extra_instructions or "").strip(), (request.instructions or "").strip()
    ] if s)
    context_block = render_context_block(memories=memories, summary=summary, instructions=instructions)

    history = items_to_messages(history_items)
    ctx = AgentContext(
        system_prompt=system_prompt,
        history=history,
        current_turn=_input_to_turn(request.input),
        attachments=[],
        hints=[],
        tools=toolbox,
        run_vars=RunVars(),
        context_block=context_block,
    )
    return ctx, conversation_id
