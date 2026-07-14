from __future__ import annotations
import asyncio
import time
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from loguru import logger

from agent.context import AgentContext, RunVars
from agent.custom_skills import (
    _normalize_skill_id,
    discover_skill_packages,
    render_skill_catalog,
    resolve_skill_mounts,
    skill_mount_fingerprint,
    skill_sources,
)
from agent.message import Message, ToolCall
from agent.tools.base import ToolBox
from agent.tools.registry import ToolRegistry
from agent.tools.builtin.spawn_subagent import SPAWN_TOOL_NAMES
from agent.tools.knowledge_bundle import KNOWLEDGE_TOOL_NAMES, normalize_knowledge_tool_lists
from agent.tools.scope import ToolScope
from app.schemas import ResponsesRequest
from app.store.base import Item, new_conversation_id
from agent.soul import (
    DEFAULT_INSTRUCTIONS,
    render_context_block,
    render_stable_system_prompt,
    render_subagent_system_prompt,
)

MEMORY_INJECT_LIMIT = 30


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


async def _async_none():
    # A ready awaitable yielding None, so asyncio.gather() slots can be filled
    # unconditionally (a skipped read contributes None instead of a branch).
    return None


async def build_context(
    request: ResponsesRequest,
    store,
    *,
    registry: Optional[ToolRegistry] = None,
    agent_config=None,
    project_context: str = "",
    authenticated_user_id: Optional[str] = None,
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store and assemble the AgentContext.
    The selected agent's ``instructions`` (a single freeform Markdown document,
    falling back to ``DEFAULT_INSTRUCTIONS`` when blank) IS the stable base system
    prompt — no org-persona merge. Selects tools from the registry, narrowed by the
    agent profile's include/exclude; registry=None keeps empty-ToolBox behavior.
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
                registry, agent_profile,
                force=(_SKILL_LOADER_TOOLS if skills_active else ()) + _CONTEXT_RECOVERY_TOOLS,
            )
        )
    else:
        toolbox = ToolBox([])

    tool_names = [t.name for t in toolbox.tools]

    code_config = getattr(agent_profile, "code", None)
    code_enabled = bool(getattr(code_config, "enabled", False))
    code_manifest = getattr(code_config, "manifest", "") or ""
    # The agent's ``instructions`` markdown IS the persona (base system prompt);
    # blank falls back to the built-in DEFAULT_INSTRUCTIONS.
    instructions_md = (getattr(agent_profile, "instructions", "") or "").strip() or DEFAULT_INSTRUCTIONS
    system_prompt = render_stable_system_prompt(
        instructions_md, tool_names=tool_names, project_context=project_context,
        aliyun_pai_enabled=_aliyun_pai_enabled(),
        code_enabled=code_enabled, code_manifest=code_manifest,
    )

    uid = authenticated_user_id or request.resolved_user_id

    # Independent per-turn store reads run concurrently (each store op uses its
    # own session/connection) instead of a serial await chain: the conversation
    # (for summary), the user's injected memories, and the user row — loaded once
    # here and threaded into the aliyun resolvers so they don't each re-fetch it.
    conv, memory_rows, user = await asyncio.gather(
        store.get_conversation(conversation_id) if conversation_id else _async_none(),
        store.list_memories(uid, limit=MEMORY_INJECT_LIMIT) if uid else _async_none(),
        store.get_user(uid) if uid else _async_none(),
    )

    summary = ""
    if conv is not None and conv.summary:
        summary = conv.summary
        history_items = [it for it in history_items if it.seq > conv.summarized_seq]

    memories: List[str] = [m.text for m in (memory_rows or [])]
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
    # Volatile per-turn additions only. The agent's own instructions are now the
    # stable base (above), so they are NOT repeated here; this carries the one-off
    # per-request instructions and the active skills' loaded guidance.
    instructions = "\n\n".join(
        s
        for s in [
            (request.instructions or "").strip(),
            skill_instructions.strip(),
        ]
        if s
    )
    context_block = render_context_block(memories=memories, summary=summary, instructions=instructions)

    metadata = dict(request.metadata or {})
    # Both aliyun resolvers take the already-loaded `user` (no re-fetch) and run
    # concurrently: the sandbox-env one may AssumeRole (cached, but a cold key is a
    # network hop), the flags one is cheap — no reason to serialize them.
    aliyun_env, aliyun_flags = await asyncio.gather(
        _resolve_aliyun_sandbox_env(agent_config, uid, user),
        _resolve_aliyun_flags(agent_config, uid, user),
    )
    if aliyun_env:
        metadata["aliyun_sandbox_env"] = aliyun_env
    # Flags for the reactive authorization card the shell tool surfaces when an
    # aliyun CLI call fails: whether authz is usable here at all, and whether this
    # user is already bound (drives "去授权" vs "重新校验/重新授权").
    metadata.update(aliyun_flags)
    # Per-agent knowledge soft default: knowledge_search / knowledge_find fall back to
    # these bases when the model passes no explicit kb_ids. Each is still
    # permission-checked per request downstream — this narrows, never widens.
    agent_kbs = None
    agent_rerank = None
    if agent_profile is not None:
        agent_knowledge = getattr(agent_profile, "knowledge", None)
        agent_kbs = getattr(agent_knowledge, "kb_ids", None)
        if agent_kbs:
            metadata["default_kb_ids"] = list(agent_kbs)
        agent_rerank = getattr(agent_knowledge, "rerank", None)
        if agent_rerank is not None:
            metadata["knowledge_rerank"] = agent_rerank.model_dump()
    logger.debug(
        "agent knowledge tools resolved: agent_id={} tools={} kb_ids={} rerank_enabled={}",
        _agent_id(agent_config, agent_profile),
        [name for name in tool_names if name in KNOWLEDGE_TOOL_NAMES],
        list(agent_kbs or []),
        bool(agent_rerank and agent_rerank.enabled),
    )

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


def build_subagent_context(
    *,
    profile,
    task: str,
    parent_scope: ToolScope,
    registry: ToolRegistry,
    agent_config=None,
    project_context: str = "",
    depth: int = 1,
) -> AgentContext:
    """Assemble a CLEAN AgentContext for a delegated subagent — the context firewall.

    Unlike ``build_context`` this does NOT touch the store: history is empty, and no
    user memory or rolling summary is injected. The subagent gets its own persona
    (``profile.instructions``) and toolbox, minus the spawn tools so it can't nest.
    It inherits the parent's user/conversation/metadata scope so sandbox mounts and
    KB permissions stay correct, but runs in a fresh window — its noisy exploration
    stays here; only its final summary returns to the caller."""
    skill_packages = _skill_packages(agent_config)
    enabled_skill_ids = _enabled_skill_ids(agent_config, profile)
    skills_active = bool(skill_packages and enabled_skill_ids)

    tool_names = _select_tool_names(
        registry, profile,
        force=(_SKILL_LOADER_TOOLS if skills_active else ()) + _CONTEXT_RECOVERY_TOOLS,
    )
    # Depth cap = 1: a subagent never gets the spawn tools, so it cannot nest.
    tool_names = [n for n in tool_names if n not in SPAWN_TOOL_NAMES]
    toolbox = registry.build_toolbox(tool_names)
    effective_names = [t.name for t in toolbox.tools]

    code_config = getattr(profile, "code", None)
    instructions_md = (getattr(profile, "instructions", "") or "").strip() or DEFAULT_INSTRUCTIONS
    system_prompt = render_subagent_system_prompt(
        instructions_md, tool_names=effective_names, project_context=project_context,
        aliyun_pai_enabled=_aliyun_pai_enabled(),
        code_enabled=bool(getattr(code_config, "enabled", False)),
        code_manifest=getattr(code_config, "manifest", "") or "",
    )

    task_turn = Message(role="user", content=task)
    skill_instructions = (
        _active_skill_instructions(
            packages=skill_packages, enabled_ids=enabled_skill_ids, current_turn=task_turn,
        )
        if skills_active
        else ""
    )
    context_block = render_context_block(instructions=skill_instructions) if skill_instructions else ""
    skill_mounts = (
        resolve_skill_mounts(
            packages=skill_packages,
            enabled_ids=enabled_skill_ids,
            skill_config=getattr(agent_config, "skills", None),
        )
        if agent_config is not None and skills_active
        else []
    )

    # Inherit the parent's runtime scope (sandbox creds, aliyun env, admin flags),
    # but stamp the subagent depth and swap the KB soft-default to the child's own.
    metadata = dict(parent_scope.metadata or {})
    metadata["subagent_depth"] = depth
    kb_ids = getattr(getattr(profile, "knowledge", None), "kb_ids", None)
    if kb_ids:
        metadata["default_kb_ids"] = list(kb_ids)
    else:
        metadata.pop("default_kb_ids", None)  # e.g. explore searches every accessible KB
    rerank = getattr(getattr(profile, "knowledge", None), "rerank", None)
    if rerank is not None:
        metadata["knowledge_rerank"] = rerank.model_dump()
    else:
        metadata.pop("knowledge_rerank", None)

    return AgentContext(
        system_prompt=system_prompt,
        history=[],
        current_turn=task_turn,
        attachments=[],
        hints=[],
        tools=toolbox,
        run_vars=RunVars(),
        context_block=context_block,
        user_id=parent_scope.user_id,
        conversation_id=parent_scope.conversation_id,
        metadata=metadata,
        agent_id=getattr(profile, "id", "subagent") or "subagent",
        skill_mounts=[mount.to_dict() for mount in skill_mounts],
        skill_fingerprint=skill_mount_fingerprint(skill_mounts),
    )


def _capability_enabled(agent_config, cap_id: str) -> bool:
    """A capability is "on" unless explicitly permission="disabled". The old global
    ``enabled`` boolean is a deprecated no-op; provider presence + per-agent tool
    selection decide actual use."""
    for cap in (getattr(agent_config, "capabilities", []) or []):
        if getattr(cap, "id", "") == cap_id:
            return getattr(cap, "permission", "") != "disabled"
    return False


def _aliyun_pai_enabled() -> bool:
    """Master switch for the PAI authorization feature (env ALIYUN_PAI_ENABLED,
    default on). Replaces the old per-config `aliyun_pai` capability toggle, which
    could persist as false and silently shadow a fully-wired deployment."""
    from app.config import get_settings
    return bool(get_settings().aliyun_pai_enabled)


async def _resolve_aliyun_flags(agent_config, uid, user) -> dict:
    """Cheap booleans (no AssumeRole) for the reactive authorization card.

    ``aliyun_authz_available`` = the PAI feature is on (ALIYUN_PAI_ENABLED) AND the
    deployment can actually run an authorization (HMAC secret, developer base
    AK/SK, and a resolvable ROS template — explicit URL or a self-hostable
    developer account id). ``aliyun_bound`` = this user has a stored binding,
    even if minting creds later fails (e.g. a deleted role) — so the card can
    offer "re-verify" instead of a fresh "authorize". Returns ``{}`` when authz
    isn't available, so the shell tool never surfaces a card that can't work.

    ``user`` is the already-loaded row from build_context — no re-fetch here.
    """
    if not uid or not _aliyun_pai_enabled():
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
        binding = (user.meta or {}).get("aliyun_pai") if user else None
        return {
            "aliyun_authz_available": True,
            "aliyun_bound": bool(binding and binding.get("role_arn")),
        }
    except Exception as exc:  # never break context build over a UX hint
        logger.warning("aliyun authz flags unavailable: {}", exc)
        return {}


# Per-user STS session env cache, keyed by (uid, role_arn, region). AssumeRole is
# a network round-trip on the first-token path; the minted token lives ~1h, so we
# reuse it until it's within _STS_REFRESH_MARGIN_S of expiry rather than calling
# STS every turn. The sandbox provider independently re-injects fresh creds into a
# long-lived cached sandbox before ALIBABACLOUD_SESSION_EXPIRATION, so a cache hand
# out is always well inside the token's validity thanks to the margin.
_STS_ENV_CACHE: dict = {}
_STS_REFRESH_MARGIN_S = 300


def _parse_iso_expiry(iso: str) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001 — unparsable expiry just means "don't cache"
        return None


async def _resolve_aliyun_sandbox_env(agent_config, uid, user) -> dict:
    """Best-effort per-user Aliyun session env for the sandbox.

    If the user has authorized PAI access (a stored role binding) and the
    feature is on (ALIYUN_PAI_ENABLED), AssumeRole to mint temp creds and return
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
    if not _aliyun_pai_enabled():
        return _skip("ALIYUN_PAI_ENABLED is off")
    try:
        from agent.integrations import aliyun_sts
        from app.config import get_settings

        settings = get_settings()
        if not settings.aliyun_authz_secret:
            return _skip("aliyun_authz_secret not configured")
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
        # Reuse a still-valid minted session instead of hitting STS on this turn's
        # first-token path. Keyed by role_arn+region so a re-authorization (new role)
        # or region change misses and re-mints.
        cache_key = (uid, binding["role_arn"], default_region)
        cached = _STS_ENV_CACHE.get(cache_key)
        if cached is not None and time.time() < cached[0] - _STS_REFRESH_MARGIN_S:
            return dict(cached[1])
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
        exp_ts = _parse_iso_expiry(creds.expiration)
        if exp_ts is not None:
            _STS_ENV_CACHE[cache_key] = (exp_ts, dict(env))
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
_SKILL_LOADER_TOOLS = ("load_skill",)

# read_handle recovers tool results the budget compressor offloaded from the window.
# It's a system recovery tool: force it past any include-whitelist/exclude so an
# offloaded placeholder always has a working way to be re-read (when registered).
_CONTEXT_RECOVERY_TOOLS = ("read_handle",)


def _select_tool_names(registry, agent_profile, *, force: Iterable[str] = ()) -> List[str]:
    """Effective toolbox = every registered tool, then narrowed by the agent
    profile's include/exclude. ``include`` (when non-empty) restricts to that set;
    ``exclude`` always subtracts. No profile → all registered tools.

    ``force`` names tools a capability requires that must survive the include/exclude
    filter (e.g. the skill loaders, which pair with an injected catalog) — added when
    actually registered, so a catalog never advertises a filtered-out tool. Appended
    last, preserving the configured selection's order."""
    available = registry.names()
    base = available
    tools_cfg = getattr(agent_profile, "tools", None) if agent_profile is not None else None
    include = list(getattr(tools_cfg, "include", []) or [])
    exclude_list = list(getattr(tools_cfg, "exclude", []) or [])
    include, exclude_list = normalize_knowledge_tool_lists(include, exclude_list)
    exclude = set(exclude_list)
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
    """The agent's enabled skills that are actually usable: intersect the profile's
    ``skills.enabled`` with installed skills whose status is ready. Skills live in
    ``skills.installed`` (no longer capabilities); a skill enabled for the agent but
    absent from ``installed`` (or not ready) is dropped. When no record exists at all
    (e.g. a disk-only test fixture), assume ready so mounting still works."""
    if agent_config is None or agent_profile is None:
        return []
    installed = getattr(getattr(agent_config, "skills", None), "installed", None) or []
    status_by_id = {}
    for rec in installed:
        rid = rec.get("id") if isinstance(rec, dict) else getattr(rec, "id", None)
        if not rid:
            continue
        status = rec.get("status") if isinstance(rec, dict) else getattr(rec, "status", "ready")
        status_by_id[_normalize_skill_id(str(rid))] = status
    result: List[str] = []
    for skill_id in getattr(agent_profile.skills, "enabled", []) or []:
        status = status_by_id.get(_normalize_skill_id(str(skill_id)), "ready")
        if status in {"ready", "untested"}:
            result.append(skill_id)
    return result


def _active_skill_instructions(*, packages: list, enabled_ids: List[str], current_turn: Message) -> str:
    """Level-1 progressive disclosure: always inject the catalog (name +
    description + id) of the agent's enabled skills so it knows which skills it
    has and when to reach for one. Full instructions (L2) are loaded on demand by
    the agent via load_skill; bundled files are accessed through the sandbox mount.
    Nothing is preloaded into every turn or gated on a lexical query match (which
    failed cross-language and for trigger-less community skills)."""
    if not packages or not enabled_ids:
        return ""
    return render_skill_catalog(packages=packages, enabled_ids=enabled_ids)
