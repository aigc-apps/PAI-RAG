from __future__ import annotations
import asyncio
from typing import Iterable, List, Optional, Tuple

from loguru import logger

MEMORY_INJECT_LIMIT = 30
from agent.context import AgentContext, RunVars
from agent.custom_skills import (
    discover_skill_packages,
    render_skill_catalog,
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
    authenticated_user_id: Optional[str] = None,
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store and assemble the AgentContext.
    Composes the effective soul (default <- request.soul <- instructions) into a
    layered system prompt. Selects tools from the registry governed by
    soul.tools_enabled; registry=None keeps empty-ToolBox behavior.
    Raises ValueError on previous_response_id/conversation conflict (-> HTTP 400).

    ``authenticated_user_id`` (the JWT ``sub``) is the source of truth for
    identity — memory injection, the per-user Aliyun sandbox creds, and
    ``ctx.user_id`` all use it, so a client can never impersonate another user by
    stuffing a ``user_id`` into the request body. Falls back to the request's
    self-declared id only when unauthenticated (e.g. internal callers)."""
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

    # Resolve the selected agent profile up front — it steers the persona name,
    # the toolbox, and the extra instructions below. None when no agent_config
    # (unit tests) or no matching profile, in which case behavior is unchanged.
    agent_profile = _resolve_agent_profile(agent_config, request)

    override = dict(request.soul or {})
    if agent_profile is not None and getattr(agent_profile, "name", ""):
        override.setdefault("name", agent_profile.name)
    effective_soul = soul.merge(override)

    # The skill catalog (L1, injected below) and the skill-loader tools are the two
    # halves of progressive disclosure and must be coupled: whenever the catalog is
    # present (the agent has enabled, discovered skills) the model is told to call
    # load_skill, so those tools have to be in the toolbox even under an
    # include-whitelist that lists only the agent's domain tools. Computed here so the
    # force-set can be threaded into tool selection.
    skill_packages = _skill_packages(agent_config)
    enabled_skill_ids = _enabled_skill_ids(agent_config, agent_profile)
    skills_active = bool(skill_packages and enabled_skill_ids)

    if registry is not None:
        toolbox = registry.build_toolbox(
            _select_tool_names(
                registry, effective_soul, agent_profile,
                force=_SKILL_LOADER_TOOLS if skills_active else (),
            )
        )
    else:
        toolbox = ToolBox([])

    tool_names = [t.name for t in toolbox.tools]

    system_prompt = render_stable_system_prompt(
        effective_soul, tool_names=tool_names, project_context=project_context,
        aliyun_pai_enabled=_aliyun_pai_enabled(agent_config),
    )

    summary = ""
    if conversation_id:
        conv = await store.get_conversation(conversation_id)
        if conv is not None and conv.summary:
            summary = conv.summary
            history_items = [it for it in history_items if it.seq > conv.summarized_seq]

    memories: List[str] = []
    uid = authenticated_user_id or request.resolved_user_id
    if uid:
        memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)]
    current_turn = _input_to_turn(request.input)
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
    profile_instructions = (getattr(agent_profile, "instructions", "") or "").strip() if agent_profile else ""
    instructions = "\n\n".join(
        s
        for s in [
            (effective_soul.extra_instructions or "").strip(),
            profile_instructions,
            (request.instructions or "").strip(),
            skill_instructions.strip(),
        ]
        if s
    )
    context_block = render_context_block(memories=memories, summary=summary, instructions=instructions)

    metadata = dict(request.metadata or {})
    aliyun_env = await _resolve_aliyun_sandbox_env(store, agent_config, uid)
    if aliyun_env:
        metadata["aliyun_sandbox_env"] = aliyun_env
    # Flags for the reactive authorization card the shell tool surfaces when an
    # aliyun CLI call fails: whether authz is usable here at all, and whether this
    # user is already bound (drives "去授权" vs "重新校验/重新授权").
    metadata.update(await _resolve_aliyun_flags(store, agent_config, uid))

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
        user_id=uid,
        conversation_id=conversation_id,
        metadata=metadata,
        agent_id=_agent_id(agent_config, agent_profile),
        skill_mounts=[mount.to_dict() for mount in skill_mounts],
        skill_fingerprint=skill_mount_fingerprint(skill_mounts),
    )
    return ctx, conversation_id


def _capability_enabled(agent_config, cap_id: str) -> bool:
    for cap in (getattr(agent_config, "capabilities", []) or []):
        if getattr(cap, "id", "") == cap_id:
            return bool(getattr(cap, "enabled", False))
    return False


def _aliyun_pai_enabled(agent_config) -> bool:
    return _capability_enabled(agent_config, "aliyun_pai")


async def _resolve_aliyun_flags(store, agent_config, uid) -> dict:
    """Cheap booleans (no AssumeRole) for the reactive authorization card.

    ``aliyun_authz_available`` = the aliyun_pai capability is enabled AND the
    deployment can actually run an authorization (HMAC secret, developer base
    AK/SK, and a resolvable ROS template — explicit URL or a self-hostable
    developer account id). ``aliyun_bound`` = this user has a stored binding,
    even if minting creds later fails (e.g. a deleted role) — so the card can
    offer "re-verify" instead of a fresh "authorize". Returns ``{}`` when authz
    isn't available, so the shell tool never surfaces a card that can't work.
    """
    if not uid or not _aliyun_pai_enabled(agent_config):
        return {}
    try:
        from agent.integrations import aliyun_sts
        from app.config import get_settings

        settings = get_settings()
        base_ak, base_sk = aliyun_sts.read_base_creds(aliyun_sts.provider_settings(agent_config))
        available = bool(
            settings.aliyun_authz_secret
            and base_ak and base_sk
            and (settings.aliyun_ros_template_url or settings.aliyun_developer_account_id)
        )
        if not available:
            return {}
        user = await store.get_user(uid)
        binding = (user.meta or {}).get("aliyun_pai") if user else None
        return {
            "aliyun_authz_available": True,
            "aliyun_bound": bool(binding and binding.get("role_arn")),
        }
    except Exception as exc:  # never break context build over a UX hint
        logger.warning("aliyun authz flags unavailable: {}", exc)
        return {}


async def _resolve_aliyun_sandbox_env(store, agent_config, uid) -> dict:
    """Best-effort per-user Aliyun session env for the sandbox.

    If the user has authorized PAI access (a stored role binding) and the
    capability is enabled, AssumeRole to mint temp creds (up to 12h) and return
    the three ALIBABACLOUD_* env vars plus region hints and the token expiry.
    NEVER raises — any failure (no binding, expired trust, missing CLI/creds)
    returns {} so sandbox creation is unaffected. The sandbox provider re-injects
    fresh creds before ALIBABACLOUD_SESSION_EXPIRATION so long sessions never see
    an expired token.
    """
    # An empty result means the sandbox gets no aliyun creds and its CLI reports
    # "profile default is not configure yet" — invisible in the sandbox request logs
    # (the env contract simply lacks ALIBABACLOUD_*), so name the reason here. Only
    # log when a sandbox actually exists to inject into: a no-sandbox deployment has
    # nothing to configure, so staying silent there avoids per-turn noise while never
    # hiding a reason from a deployment that could run aliyun.
    sandbox_on = _capability_enabled(agent_config, "sandbox")

    def _skip(reason: str) -> dict:
        if sandbox_on:
            logger.info("aliyun sandbox env: no creds for user={} ({})", uid or "-", reason)
        return {}

    if not uid:
        return _skip("no user id on this turn")
    if not _aliyun_pai_enabled(agent_config):
        return _skip("aliyun_pai capability disabled")
    try:
        from agent.integrations import aliyun_sts
        from app.config import get_settings

        settings = get_settings()
        if not settings.aliyun_authz_secret:
            return _skip("aliyun_authz_secret not configured")
        user = await store.get_user(uid)
        binding = (user.meta or {}).get("aliyun_pai") if user else None
        if not binding or not binding.get("role_arn"):
            return _skip("user has no PAI authorization binding (not authorized, "
                         "or authorization did not persist a role_arn)")
        pai_settings = aliyun_sts.provider_settings(agent_config)
        base_ak, base_sk = aliyun_sts.read_base_creds(pai_settings)
        if not (base_ak and base_sk):
            return _skip("developer base AK/SK not configured")
        # STS AssumeRole is region-agnostic (endpoint selection only); the minted
        # token works in every region. Prefer the binding's default/service region,
        # falling back to legacy single-region bindings, then the configured region.
        default_region = (
            binding.get("default_region")
            or binding.get("region")
            or aliyun_sts.configured_region(pai_settings, settings.aliyun_default_region)
        )
        creds = await asyncio.to_thread(
            aliyun_sts.assume_role,
            binding["role_arn"],
            binding.get("external_id")
            or aliyun_sts.derive_external_id(
                uid,
                secret=settings.aliyun_authz_secret,
                prefix=aliyun_sts.configured_external_id_prefix(pai_settings),
            ),
            region=default_region,
            base_ak=base_ak, base_sk=base_sk,
            account_id=settings.aliyun_developer_account_id,
            duration_seconds=aliyun_sts.configured_assume_duration_seconds(pai_settings),
        )
        env = aliyun_sts.to_sandbox_env(creds)
        # Region hints for the agent: a sane default plus the discovered regions so
        # it knows where the customer's PAI services live (token is valid in all).
        env["ALIBABACLOUD_REGION_ID"] = default_region
        available = binding.get("service_regions") or binding.get("regions") or []
        if available:
            env["PAI_AVAILABLE_REGIONS"] = ",".join(available)
        # Token expiry (ISO8601) so the sandbox provider can re-inject fresh creds
        # into a long-lived cached sandbox before this token expires.
        if creds.expiration:
            env["ALIBABACLOUD_SESSION_EXPIRATION"] = creds.expiration
        logger.info(
            "aliyun sandbox env: minted STS creds for user={} region={} expires={}",
            uid, default_region, creds.expiration or "?",
        )
        return env
    except Exception as exc:  # noqa: BLE001 — injection is strictly best-effort
        logger.warning("aliyun sandbox env skipped for user={}: {}", uid, exc)
        return {}


def _resolve_agent_profile(agent_config, request: ResponsesRequest):
    if agent_config is None:
        return None
    agents = getattr(agent_config, "agents", []) or []
    requested_id = request.agent_id or (request.metadata or {}).get("agent_id")
    agent_id = requested_id or getattr(agent_config, "default_agent", "main")
    return next((item for item in agents if item.id == agent_id), agents[0] if agents else None)


# The execute half of progressive disclosure. Coupled to catalog injection in
# build_context: forced into the toolbox iff the agent has active skills, so an
# include-whitelist or exclude can't strip the tool the catalog tells the model to
# call. See _select_tool_names(force=...).
_SKILL_LOADER_TOOLS = ("load_skill", "read_skill_resource")


def _select_tool_names(registry, soul, agent_profile, *, force: Iterable[str] = ()) -> List[str]:
    """Effective toolbox = the soul's base selection, then narrowed by the agent
    profile's include/exclude. ``include`` (when non-empty) restricts to that set;
    ``exclude`` always subtracts. No profile → unchanged soul/registry behavior.

    ``force`` names tools a capability requires that must survive the include/exclude
    filter (e.g. the skill loaders, which pair with an injected catalog) — added when
    actually registered, so a catalog never advertises a filtered-out tool. Appended
    last, preserving the configured selection's order."""
    available = registry.names()
    base = soul.tools_enabled if soul.tools_enabled is not None else available
    tools_cfg = getattr(agent_profile, "tools", None) if agent_profile is not None else None
    include = list(getattr(tools_cfg, "include", []) or [])
    exclude = set(getattr(tools_cfg, "exclude", []) or [])
    if include:
        base = [n for n in include if n in available]
    selected = [n for n in base if n not in exclude]
    for name in force:
        if name in available and name not in selected:
            selected.append(name)
    return selected


def resolve_agent_model(agent_config, request: ResponsesRequest) -> Optional[str]:
    """The model the selected agent profile pins, if any — used by the route to
    default ``request.model`` before the client's own choice/override."""
    profile = _resolve_agent_profile(agent_config, request)
    if profile is not None:
        return (getattr(profile, "model", "") or "") or None
    return None


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
    """Level-1 progressive disclosure: always inject the catalog (name +
    description + id) of the agent's enabled skills so it knows which skills it
    has and when to reach for one. Full instructions (L2) and bundled files (L3)
    are loaded on demand by the agent via the load_skill / read_skill_resource
    tools — never preloaded into every turn, and never gated on a lexical query
    match (which failed cross-language and for trigger-less community skills)."""
    if not packages or not enabled_ids:
        return ""
    return render_skill_catalog(packages=packages, enabled_ids=enabled_ids)
