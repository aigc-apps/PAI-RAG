"""``@function_tool`` definitions for every legacy ``do_*`` handler.

Each wrapper is intentionally thin: flat keyword args → ``handler.do_<name>``
→ JSON-serializable result. ``ctx.context`` is a
:class:`backend.agents_sdk.runner.RunContext`; the runner registers the
per-run ``GenericHandler`` in a process-local table (``runner._RUN_EXTRAS``)
keyed by ``run_id``, and these wrappers look it up there. The handler is
not stored on ``RunContext`` directly because ``RunState.to_string()``
pickles the context and the handler holds un-pickleable locks.

Parameters are flattened (top-level kwargs, not nested under a Pydantic
wrapper). A nested ``{args: {...}}`` schema confuses LLMs — they emit
flat ``{skill, args}`` and the SDK rejects with `model_type` / `missing`
validation errors. Top-level kwargs match the LLM's natural output and
also produce a cleaner schema.

``ask_user`` uses ``needs_approval=True``. The handler body returns whatever
text the user provides via the resume path — the runner populates
``ctx.context.pending_human_answer`` before the SDK re-invokes the tool.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from agents import RunContextWrapper, function_tool


# ────────────────────────── helpers ──────────────────────────

def _handler(ctx):
    """Pull the per-run ``GenericHandler`` from the runner's process-local
    side table (keyed by ``run_id``).

    The handler holds threading.RLock instances and other un-pickleable
    objects, so it can't live on ``RunContext`` directly — ``RunState``
    serialization would crash on completion or HITL pause.
    """
    from backend.agents_sdk.runner import get_run_extras
    run_id = getattr(getattr(ctx, 'context', None), 'run_id', '') or ''
    handler = get_run_extras(run_id).get('handler')
    if handler is None:
        raise RuntimeError(
            f'no GenericHandler registered for run_id={run_id!r}; '
            f'runner must call register_run_extras before invoking tools'
        )
    return handler


def _outcome_to_str(outcome: Any) -> str:
    """SDK function tools expect a string return. We JSON-encode the legacy
    ``StepOutcome.data`` dict; the LLM sees structured output it can parse.
    Strings pass through unchanged for human-readable tool results.

    The legacy loop also injects ``StepOutcome.next_prompt`` as a follow-up
    user message — for ``use_skill`` this carries the actual skill body
    (``[SKILL ACTIVATED: ...]\\n<instructions>``), without which the agent
    only sees ``{status: skill_activated}`` and prematurely terminates with
    ``<summary>``. The SDK has no separate follow-up channel, so we
    concatenate ``next_prompt`` into the tool-result string.
    """
    data = getattr(outcome, 'data', outcome)
    if isinstance(data, str):
        body = data
    else:
        try:
            body = json.dumps(data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            body = str(data)
    next_prompt = getattr(outcome, 'next_prompt', None)
    if next_prompt and str(next_prompt).strip():
        return f'{body}\n\n{next_prompt}'
    return body


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ────────────────────────── wrappers ──────────────────────────

@function_tool(name_override='file_read', strict_mode=False)
async def file_read(
    ctx: RunContextWrapper[Any],
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    payload = _drop_none({'path': path, 'start_line': start_line, 'end_line': end_line})
    return _outcome_to_str(_handler(ctx).do_file_read(payload, None))


@function_tool(name_override='file_write', strict_mode=False)
async def file_write(
    ctx: RunContextWrapper[Any],
    path: str,
    content: str,
    mode: Optional[str] = 'overwrite',
) -> str:
    payload = _drop_none({'path': path, 'content': content, 'mode': mode})
    return _outcome_to_str(_handler(ctx).do_file_write(payload, None))


@function_tool(name_override='file_patch', strict_mode=False)
async def file_patch(
    ctx: RunContextWrapper[Any],
    path: str,
    patch: str,
) -> str:
    return _outcome_to_str(_handler(ctx).do_file_patch({'path': path, 'patch': patch}, None))


@function_tool(name_override='code_run', strict_mode=False)
async def code_run(
    ctx: RunContextWrapper[Any],
    script: str,
    type: Optional[str] = 'python',
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """Execute a script (python or bash) and return its result.

    Parameter names mirror the legacy ``do_code_run`` schema (``script``,
    ``type``, ``cwd``, ``timeout``) so the handler actually finds the code.
    Earlier this wrapper used ``code``/``language``, which the handler
    silently ignored and then crashed dereferencing ``response.content``
    (response is None on the SDK path) — surface to the user as
    ``'NoneType' object has no attribute 'content'``.
    """
    payload = _drop_none({'script': script, 'type': type, 'cwd': cwd, 'timeout': timeout})
    return _outcome_to_str(_handler(ctx).do_code_run(payload, None))


@function_tool(name_override='ask_user', needs_approval=True, strict_mode=False)
async def ask_user(
    ctx: RunContextWrapper[Any],
    question: str,
    candidates: Optional[list[str]] = None,
    reason: Optional[str] = '',
    risk: Optional[str] = '',
    default_action: Optional[str] = '',
) -> str:
    """Pause and ask the user a clarifying question.

    By default the runner may autonomously approve this tool with
    ``default_action`` (or a conservative fallback) instead of pausing the
    frontend. When the request explicitly enables HITL, the body only runs
    after :meth:`RunState.approve` is called on the corresponding
    ``ToolApprovalItem``. The runner stashes the human reply on
    ``ctx.context.pending_human_answer`` before resuming, so this body is a
    trivial echo.
    """
    answer = getattr(ctx.context, 'pending_human_answer', '') or ''
    return answer


@function_tool(name_override='update_working_checkpoint', strict_mode=False)
async def update_working_checkpoint(
    ctx: RunContextWrapper[Any],
    key_info: Optional[str] = None,
    related_sop: Optional[str] = None,
) -> str:
    payload = _drop_none({'key_info': key_info, 'related_sop': related_sop})
    return _outcome_to_str(_handler(ctx).do_update_working_checkpoint(payload, None))


@function_tool(name_override='update_todo', strict_mode=False)
async def update_todo(
    ctx: RunContextWrapper[Any],
    items: list[dict],
) -> str:
    return _outcome_to_str(_handler(ctx).do_update_todo({'items': items}, None))


@function_tool(name_override='use_skill', strict_mode=False)
async def use_skill(
    ctx: RunContextWrapper[Any],
    skill: str,
    args: Optional[str] = '',
) -> str:
    return _outcome_to_str(_handler(ctx).do_use_skill({'skill': skill, 'args': args or ''}, None))


@function_tool(name_override='final_report', strict_mode=False)
async def final_report(
    ctx: RunContextWrapper[Any],
    report_markdown: str,
    summary: Optional[str] = '',
    evidence_refs: Optional[list[str]] = None,
) -> str:
    """Submit the structured, user-visible final answer for this run.

    The SDK runner treats this as a terminal contract: it captures the report
    out-of-band and emits ``report_markdown`` as the final assistant message.
    """
    from backend.agents_sdk.runner import set_run_extra

    report = (report_markdown or '').strip()
    if not report:
        report = (summary or '').strip() or 'Final report was submitted without report_markdown.'
    payload = {
        'report_markdown': report,
        'summary': (summary or '').strip(),
        'evidence_refs': evidence_refs or [],
    }
    run_id = getattr(getattr(ctx, 'context', None), 'run_id', '') or ''
    set_run_extra(run_id, 'final_report', payload)
    return json.dumps(
        {
            'status': 'final_report_accepted',
            'message': 'Final report captured; report_markdown will be delivered as the user-visible final answer.',
            'summary': payload['summary'],
            'evidence_refs': payload['evidence_refs'],
        },
        ensure_ascii=False,
    )


@function_tool(name_override='start_long_term_update', strict_mode=False)
async def start_long_term_update(ctx: RunContextWrapper[Any]) -> str:
    return _outcome_to_str(_handler(ctx).do_start_long_term_update({}, None))


# ────────────────────────── tool sets ──────────────────────────

# Mirrors the legacy filtering in backend/tool_schemas.py: foreground runs
# exclude start_long_term_update; the background memory review uses only the
# file tools + start_long_term_update.

_ALL = [
    file_read, file_write, file_patch, code_run,
    ask_user,
    update_working_checkpoint, update_todo,
    use_skill,
    final_report,
    start_long_term_update,
]

_EXCLUDE_MAIN = {'start_long_term_update'}
_BACKGROUND_NAMES = {'start_long_term_update', 'file_read', 'file_patch', 'file_write'}

TOOLS = [t for t in _ALL if t.name not in _EXCLUDE_MAIN]


def build_tool_list(*, scope: str = 'main') -> list:
    """Return the function-tool list for the given scope.

    - ``main``: foreground agent (excludes background-only tools).
    - ``background_memory_review``: only file tools + start_long_term_update.
    - ``all``: every tool (for tests).
    """
    if scope == 'main':
        return list(TOOLS)
    if scope == 'background_memory_review':
        return [t for t in _ALL if t.name in _BACKGROUND_NAMES]
    if scope == 'all':
        return list(_ALL)
    raise ValueError(f'unknown tool scope: {scope!r}')
