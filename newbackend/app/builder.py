from __future__ import annotations
from typing import List, Optional, Tuple

MEMORY_INJECT_LIMIT = 30
from agent.context import AgentContext, RunVars
from agent.custom_skills import (
    discover_skill_packages,
    render_skill_instructions,
    resolve_skill_mounts,
    skill_mount_fingerprint,
    skill_sources,
)
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
    agent_config=None,
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
    current_turn = _input_to_turn(request.input)
    agent_profile = _resolve_agent_profile(agent_config, request)
    enabled_skill_ids = _enabled_skill_ids(agent_config, agent_profile)
    skill_packages = _skill_packages(agent_config)
    skill_instructions = _active_skill_instructions(
        packages=skill_packages,
        enabled_ids=enabled_skill_ids,
        current_turn=current_turn,
    )
    skill_mounts = (
        resolve_skill_mounts(
            packages=skill_packages,
            enabled_ids=enabled_skill_ids,
            skill_config=getattr(agent_config, "skills", None),
        )
        if agent_config is not None
        else []
    )
    instructions = "\n\n".join(
        s
        for s in [
            (effective_soul.extra_instructions or "").strip(),
            (request.instructions or "").strip(),
            skill_instructions.strip(),
        ]
        if s
    )
    context_block = render_context_block(memories=memories, summary=summary, instructions=instructions)

    history = items_to_messages(history_items)
    ctx = AgentContext(
        system_prompt=system_prompt,
        history=history,
        current_turn=current_turn,
        attachments=[],
        hints=[],
        tools=toolbox,
        run_vars=RunVars(),
        context_block=context_block,
        user_id=request.resolved_user_id,
        conversation_id=conversation_id,
        metadata=dict(request.metadata or {}),
        agent_id=_agent_id(agent_config, agent_profile),
        skill_mounts=[mount.to_dict() for mount in skill_mounts],
        skill_fingerprint=skill_mount_fingerprint(skill_mounts),
    )
    return ctx, conversation_id


def _resolve_agent_profile(agent_config, request: ResponsesRequest):
    if agent_config is None:
        return None
    agents = getattr(agent_config, "agents", []) or []
    requested_id = request.agent_id or (request.metadata or {}).get("agent_id")
    agent_id = requested_id or getattr(agent_config, "default_agent", "main")
    return next((item for item in agents if item.id == agent_id), agents[0] if agents else None)


def _agent_id(agent_config, agent_profile) -> str:
    if agent_profile is not None:
        return getattr(agent_profile, "id", None) or "main"
    if agent_config is not None:
        return getattr(agent_config, "default_agent", "main")
    return "main"


def _skill_packages(agent_config) -> list:
    if agent_config is None:
        return []
    return discover_skill_packages(skill_sources(getattr(agent_config, "skills", None)))


def _enabled_skill_ids(agent_config, agent_profile) -> List[str]:
    if agent_config is None or agent_profile is None:
        return []
    caps = {
        cap.id: cap
        for cap in (getattr(agent_config, "capabilities", []) or [])
        if getattr(cap, "kind", "") == "skill"
        and getattr(cap, "enabled", False)
        and getattr(cap, "status", "ready") in {"ready", "untested"}
    }
    return [
        skill_id
        for skill_id in getattr(agent_profile.skills, "enabled", [])
        if skill_id in caps
    ]


def _active_skill_instructions(*, packages: list, enabled_ids: List[str], current_turn: Message) -> str:
    if not packages or not enabled_ids:
        return ""
    query = (
        current_turn.content
        if isinstance(current_turn.content, str)
        else str(current_turn.content or "")
    )
    return render_skill_instructions(
        packages=packages,
        enabled_ids=enabled_ids,
        query=query,
    )
