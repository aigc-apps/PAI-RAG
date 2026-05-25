"""Single entry point for running / resuming a SDK-backed agent.

The HTTP layer (``backend/server.py``) calls :func:`stream_responses_run`
(streaming) or :func:`collect_responses_run` (non-streaming); everything
underneath is hidden here.

Resume semantics:

- Caller supplies ``previous_response_id`` + a list of ``input`` items
  containing exactly one ``function_call_output`` (the user's reply).
- We load the persisted ``RunState`` blob, hydrate it via
  ``RunState.from_string``, find the matching ``ToolApprovalItem`` in
  ``state.get_interruptions()`` by ``call_id``, populate
  ``ctx.pending_human_answer`` so the ``ask_user`` tool body returns the
  right text, then ``state.approve(item)`` and re-stream via
  ``Runner.run_streamed(agent, input=state, ...)``.
"""
from __future__ import annotations

import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from agents import Runner
from agents.run_state import RunState

from backend.agents_sdk import event_bridge
from backend.agents_sdk.agent_factory import build as build_agent
from backend.agents_sdk.event_bridge import ResponsesStreamState
from backend.agents_sdk.runtime_setup import acquire_request_model
from backend.agents_sdk.hitl import (
    InterruptionEnvelope,
    ResumePayload,
    interruption_from_sdk_item,
)
from backend.agents_sdk.lifecycle import (
    RUN_STATE_COMPLETED,
    RUN_STATE_FAILED,
    RUN_STATE_REQUIRES_ACTION,
    RUN_STATE_RUNNING,
)
from backend.agents_sdk.run_state_store import RunStateStore
from backend.audit.store import (
    AuditEvent,
    AuditStore,
    CATEGORY_HITL_AUTO_CONTINUE,
    CATEGORY_HITL_PAUSE,
    CATEGORY_HITL_RESUME,
    CATEGORY_RUN_COMPLETE,
    CATEGORY_RUN_FAILED,
)


def _mint_response_id() -> str:
    return 'resp_' + secrets.token_hex(12)


def _mint_run_id() -> str:
    return 'run_' + secrets.token_hex(12)


def _mint_audit_id() -> str:
    return 'audit_' + secrets.token_hex(12)


def _mint_session_id() -> str:
    return 'sess_' + secrets.token_hex(8)


# Process-local registry for non-pickleable run-scoped objects (GenericHandler,
# session, etc.). RunState.to_string() pickles RunContext, so anything holding
# a threading.RLock would crash the run; we keep those in this side-table
# keyed by run_id and look them up from tool wrappers via ctx.context.run_id.
_RUN_EXTRAS: dict[str, dict[str, Any]] = {}
_RUN_EXTRAS_LOCK = threading.Lock()

# In-flight response_id → session_id registry. Populated on run start, cleared
# on completion. The state_store only persists rows at terminal time, so the
# cancel endpoint cannot find an in-flight response by ``load_response``;
# this registry plugs that gap. Cancel-by-response-id consults it first and
# falls back to the persisted record for already-completed runs.
_INFLIGHT_RESPONSES: dict[str, str] = {}
_INFLIGHT_LOCK = threading.Lock()


def register_inflight_response(response_id: str, session_id: str) -> None:
    if not response_id:
        return
    with _INFLIGHT_LOCK:
        _INFLIGHT_RESPONSES[response_id] = session_id or ''


def session_for_inflight_response(response_id: str) -> str | None:
    if not response_id:
        return None
    with _INFLIGHT_LOCK:
        return _INFLIGHT_RESPONSES.get(response_id)


def unregister_inflight_response(response_id: str) -> None:
    if not response_id:
        return
    with _INFLIGHT_LOCK:
        _INFLIGHT_RESPONSES.pop(response_id, None)


def register_run_extras(run_id: str, extras: dict[str, Any]) -> None:
    if not run_id:
        return
    with _RUN_EXTRAS_LOCK:
        _RUN_EXTRAS[run_id] = dict(extras or {})


def get_run_extras(run_id: str) -> dict[str, Any]:
    with _RUN_EXTRAS_LOCK:
        return dict(_RUN_EXTRAS.get(run_id, {}))


def set_run_extra(run_id: str, key: str, value: Any) -> None:
    if not run_id:
        return
    with _RUN_EXTRAS_LOCK:
        _RUN_EXTRAS.setdefault(run_id, {})[key] = value


def unregister_run_extras(run_id: str) -> None:
    if not run_id:
        return
    with _RUN_EXTRAS_LOCK:
        _RUN_EXTRAS.pop(run_id, None)


@dataclass
class RunContext:
    session_id: str
    run_id: str
    response_id: str
    user_id: str
    cwd: str
    pending_human_answer: str = ''
    allow_hitl: bool = False
    auto_hitl_count: int = 0


@dataclass
class StreamFrame:
    """One yielded item in :func:`stream_responses_run`. Either ``chunk`` is
    set (a ready-to-emit Responses-API SSE payload) or ``terminal`` is True
    (the stream ended; ``response_object`` carries the final / paused state).
    """
    chunk: dict | None = None
    terminal: bool = False
    response_object: dict | None = None
    interruption: InterruptionEnvelope | None = None


_PAIRAG_NAMESPACE_KEY = event_bridge._PAIRAG_NAMESPACE_KEY
_FINAL_REPORT_FLAG = event_bridge._FINAL_REPORT_FLAG
_PROCESS_REASONING_FLAG = event_bridge._PROCESS_REASONING_FLAG
_pairag_flag = event_bridge._pairag_flag
_pairag_metadata = event_bridge._pairag_metadata
_SUMMARY_BLOCK_RE = re.compile(r'<summary\b[^>]*>(.*?)</summary>', re.IGNORECASE | re.DOTALL)
_PRIVATE_BLOCK_RE = re.compile(
    r'<(forcing_skill_activation|clinical[-_]thinking|taking[-_]action|skill[-_]context|thinking|checking|taking|working)\b[^>]*>'
    r'.*?</\1>',
    re.IGNORECASE | re.DOTALL,
)
# Some models (observed: qwen3.6-plus) sometimes serialize the final_report
# tool call as plain assistant text — `<final_report>{"report_markdown": "..."}</final_report>` —
# instead of the proper OpenAI tool-call protocol frame. We rescue those by
# pattern-matching the wrapper post-hoc and surfacing the inner markdown via
# the normal final-report contract.
_FINAL_REPORT_TEXT_RE = re.compile(
    r'<final_report\b[^>]*>\s*(\{[\s\S]*?\})\s*</final_report>',
    re.IGNORECASE,
)
_SECRET_LINE_RE = re.compile(r'(?i)(access[_-]?key|access[_-]?id|secret|password|token)')
_AUTO_HITL_MAX_CONTINUES = 3
_AUTONOMOUS_ASK_USER_ANSWER = (
    '当前未开启用户打断。请不要等待用户输入；请基于已有证据选择最保守、可逆、低风险的默认方案继续。'
    '如果缺少用户独有信息或需要不可逆/高风险授权，请停止该动作，并在最终报告中说明阻塞原因、已验证证据和需要用户补充的信息。'
)
# default_action that is itself a meta-instruction to pause/wait creates a loop:
# the runner echoes the string back as if it were the user's answer, the model
# reads "用户回答：等待确认" and asks again, eventually hitting the max-continue
# cap. Filter those out and force the conservative answer instead.
_PAUSE_LIKE_DEFAULT_ACTION_RE = re.compile(
    r'(等待|等候|等用户|稍后|暂停|暂缓|确认|询问用户|向用户|不要继续|挂起|hold\b|wait\b|pause\b|ask\s+(?:the\s+)?user|defer\b|stand\s*by)',
    re.IGNORECASE,
)
_TOOL_INTENT_WITHOUT_CALL_RETRY_PROMPT = (
    '上一轮你在文本里声称要调用工具或激活 skill（例如「调用 ask_user」'
    '「启动/激活 X 技能」「use_skill …」「我将暂停流程」或类似表述），'
    '但本轮实际没有发起任何 tool_call，前端和执行器因此不会执行动作。\n'
    '请基于原始任务继续，并二选一：\n'
    '- 如果确实需要调用工具或激活 skill，立即真正发起对应 tool_call；\n'
    '- 如果不需要再调用工具，直接输出用户可见的最终回答正文，不要再用“我将…”之类的预告口吻。'
)
_TOOL_INTENT_WITHOUT_CALL_RE = re.compile(
    r'(?:'
    r'(?:调用|使用)\s*(?:ask_user|tool_call|tool)'
    r'|(?:向|跟|与)\s*用户\s*(?:发起|进行|做出)?\s*(?:明确)?\s*询问'
    r'|我(?:将|会|准备|打算|要|需要)\s*(?:暂停|向用户|对用户|询问用户|发起询问|调用|使用)'
    r'|因此[，,]\s*我(?:将|会|要|准备|打算)'
    r'|(?:^|[。；;，,\n]|<summary>|<taking>)\s*(?:启动|激活|开启|进入)\s*[\w\-/. ]{0,80}?\s*(?:技能|skill)'
    r'|(?:^|[。；;，,\n]|<summary>|<taking>)\s*(?:应(?:优先)?|应该|必须|立即|优先)?\s*(?:调用|使用)\s*[\w\-/. ]{0,80}?\s*(?:技能|skill)'
    r'|我\s*(?:将|会|准备|打算|要|需要)\s*(?:调用|使用)\s*[\w\-/. ]{0,80}?\s*(?:技能|skill)'
    r'|use_skill'
    r"|I\s+(?:will|am\s+going\s+to|need\s+to|have\s+to)\s+(?:call|ask|invoke|use)"
    r')',
    re.IGNORECASE,
)
_AUTONOMOUS_REJECT_ANSWER = (
    '当前未开启用户打断或人工审批。不要执行该需要审批的动作；请改用安全替代方案，'
    '或在最终报告中说明该动作被阻止以及继续所需的显式授权。'
)


def _message_output_text(item: dict) -> str:
    parts: list[str] = []
    if item.get('type') != 'message':
        return ''
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') in ('output_text', 'text'):
            text = block.get('text') or ''
            if text:
                parts.append(text)
    return ''.join(parts)


def _reasoning_output_text(item: dict) -> str:
    parts: list[str] = []
    if item.get('type') != 'reasoning':
        return ''
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') in ('reasoning_text', 'summary_text', 'text'):
            text = block.get('text') or ''
            if text:
                parts.append(text)
    return ''.join(parts)


def _is_final_report_message(item: dict) -> bool:
    return _pairag_flag(item, _FINAL_REPORT_FLAG)


def _is_process_reasoning_message(item: dict) -> bool:
    return item.get('type') == 'reasoning' or _pairag_flag(item, _PROCESS_REASONING_FLAG)


def _final_text_from_output(output: list[dict]) -> str:
    report_parts: list[str] = []
    fallback_parts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        if _is_process_reasoning_message(item):
            continue
        text = _message_output_text(item)
        if not text:
            continue
        if _is_final_report_message(item):
            report_parts.append(text)
        else:
            fallback_parts.append(text)
    return ''.join(report_parts) if report_parts else ''.join(fallback_parts)


def _final_text_after_last_tool_output(output: list[dict]) -> str:
    last_tool_output = -1
    for index, item in enumerate(output or []):
        if isinstance(item, dict) and item.get('type') == 'function_call_output':
            last_tool_output = index
    if last_tool_output < 0:
        return _final_text_from_output(output)

    report_parts: list[str] = []
    fallback_parts: list[str] = []
    for item in (output or [])[last_tool_output + 1:]:
        if not isinstance(item, dict):
            continue
        if _is_process_reasoning_message(item):
            continue
        text = _message_output_text(item)
        if not text:
            continue
        if _is_final_report_message(item):
            report_parts.append(text)
        else:
            fallback_parts.append(text)
    return ''.join(report_parts) if report_parts else ''.join(fallback_parts)


def _all_message_text_from_output(output: list[dict]) -> str:
    parts: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        text = _message_output_text(item) or _reasoning_output_text(item)
        if text:
            parts.append(text)
    return ''.join(parts)


def _final_report_from_extras(ctx: RunContext) -> str:
    report = get_run_extras(ctx.run_id).get('final_report')
    if not isinstance(report, dict):
        return ''
    return str(report.get('report_markdown') or '').strip()


def _final_report_from_text_wrapper(output: list[dict]) -> str:
    """Recover `report_markdown` from a textual `<final_report>{...}</final_report>` block.

    Some models (notably qwen3.6-plus) occasionally emit the final_report tool
    invocation as plain assistant text instead of an OpenAI tool-call frame,
    which means ``ctx.extras['final_report']`` is never populated. Walks
    output messages newest-first, returns the first parseable wrapper's
    ``report_markdown``, or '' if none match.
    """
    for item in reversed(output or []):
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        text = _message_output_text(item)
        if not text:
            continue
        match = _FINAL_REPORT_TEXT_RE.search(text)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        markdown = parsed.get('report_markdown')
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()
    return ''


def _strip_model_protocol_blocks(text: str) -> str:
    if not text:
        return ''
    text = _PRIVATE_BLOCK_RE.sub('', text)
    text = _SUMMARY_BLOCK_RE.sub('', text)
    return text


def _has_final_report_output(output: list[dict]) -> bool:
    return any(isinstance(item, dict) and _is_final_report_message(item) for item in output or [])


def _demote_plain_message_to_process_reasoning(item: dict) -> dict:
    """Rewrite a ``type=message`` item as ``type=reasoning`` with the
    ``is_process_reasoning`` flag set so naive SDK consumers (which concat
    every ``type=message`` to compute ``response.output_text``) skip it.

    Returns a new dict; caller should overwrite ``output[idx]`` with the
    return value.
    """
    rewritten_content: list[dict] = []
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') in ('output_text', 'text'):
            rewritten_content.append({'type': 'summary_text', 'text': block.get('text') or ''})
        else:
            rewritten_content.append(dict(block))
    metadata = dict(item.get('metadata') or {})
    namespace = dict(metadata.get(_PAIRAG_NAMESPACE_KEY) or {})
    namespace.pop(_FINAL_REPORT_FLAG, None)
    namespace[_PROCESS_REASONING_FLAG] = True
    metadata[_PAIRAG_NAMESPACE_KEY] = namespace
    rewritten = {
        **{k: v for k, v in item.items() if k not in ('type', 'role', 'content', 'metadata')},
        'type': 'reasoning',
        'content': rewritten_content,
        'metadata': metadata,
    }
    rewritten.pop('role', None)
    return rewritten


def _finalize_output_for_responses(state: ResponsesStreamState) -> None:
    """Enforce the single-`message` invariant in ``response.completed.output``.

    OpenAI SDK consumers compute ``response.output_text`` by concatenating every
    ``type=message`` item's text — they cannot see our ``pairag.is_final_report``
    flag. To make naive SDK usage just work, the terminal payload must contain
    at most one assistant ``message``, and that one is the canonical answer.

    With visible text now streaming live as ``output_text.delta`` (instead of
    being buffered until the final flush), pre-tool prose surfaces as a real
    ``message`` item during streaming. We demote any such message that sits
    BEFORE the last tool boundary at terminal time so the SDK consumer's naive
    concat returns only the post-tool answer.

    Mutates ``state.output`` in place. Does not emit SSE events: streaming
    clients reconcile against the terminal ``response.completed.output`` snapshot.
    """
    output = state.output
    has_tool_call = any(
        isinstance(item, dict)
        and item.get('type') in ('function_call', 'function_call_output')
        for item in output
    )
    flagged_indices = [
        i for i, item in enumerate(output)
        if isinstance(item, dict) and item.get('type') == 'message' and _is_final_report_message(item)
    ]
    plain_message_indices = [
        i for i, item in enumerate(output)
        if isinstance(item, dict) and item.get('type') == 'message' and not _is_final_report_message(item)
    ]

    if not flagged_indices and not has_tool_call and len(plain_message_indices) == 1:
        idx = plain_message_indices[0]
        item = dict(output[idx])
        metadata = dict(item.get('metadata') or {})
        namespace = dict(metadata.get(_PAIRAG_NAMESPACE_KEY) or {})
        namespace[_FINAL_REPORT_FLAG] = True
        metadata[_PAIRAG_NAMESPACE_KEY] = namespace
        item['metadata'] = metadata
        output[idx] = item
        return

    if not flagged_indices:
        if has_tool_call and plain_message_indices:
            # Find the last function_call / function_call_output index. Any
            # plain message strictly before it is pre-tool prose (we live-
            # streamed it for UX), so demote to process reasoning. Messages
            # after the last tool boundary are the model's final answer.
            last_tool_idx = max(
                i for i, item in enumerate(output)
                if isinstance(item, dict)
                and item.get('type') in ('function_call', 'function_call_output')
            )
            for idx in plain_message_indices:
                if idx < last_tool_idx:
                    item = output[idx]
                    if isinstance(item, dict):
                        output[idx] = _demote_plain_message_to_process_reasoning(item)
        return

    keep_idx = flagged_indices[-1]
    for idx in plain_message_indices + flagged_indices[:-1]:
        if idx == keep_idx:
            continue
        item = output[idx]
        if not isinstance(item, dict):
            continue
        output[idx] = _demote_plain_message_to_process_reasoning(item)


def _needs_final_report_retry(output: list[dict]) -> bool:
    if _has_final_report_output(output):
        return False
    has_tool_output = any(
        isinstance(item, dict) and item.get('type') == 'function_call_output'
        for item in output or []
    )
    if not has_tool_output:
        return False
    visible_text = _strip_model_protocol_blocks(_final_text_after_last_tool_output(output)).strip()
    return not visible_text


def _needs_tool_intent_retry(output: list[dict]) -> bool:
    has_tool_call = any(
        isinstance(item, dict) and item.get('type') in ('function_call', 'function_call_output')
        for item in output or []
    )
    if has_tool_call:
        return False
    raw_text = _all_message_text_from_output(output)
    if not raw_text:
        return False
    return bool(_TOOL_INTENT_WITHOUT_CALL_RE.search(raw_text))


def _tool_intent_retry_input(output: list[dict], original_input: str = '') -> str:
    previous = _redact_report_text(_all_message_text_from_output(output), max_chars=4000)
    task = _redact_report_text(original_input, max_chars=2000)
    sections = [_TOOL_INTENT_WITHOUT_CALL_RETRY_PROMPT]
    if task:
        sections.append(f'原始任务：\n{task}')
    if previous:
        sections.append(f'上一轮文本：\n{previous}')
    return '\n\n'.join(sections)


def _contains_cjk(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in text or '')


def _redact_report_text(text: str, *, max_chars: int = 6000) -> str:
    if not text:
        return ''
    lines: list[str] = []
    for line in text.splitlines():
        if _SECRET_LINE_RE.search(line):
            lines.append('[redacted sensitive line]')
        else:
            lines.append(line)
    redacted = '\n'.join(lines).strip()
    if len(redacted) > max_chars:
        return redacted[:max_chars].rstrip() + '\n...[truncated]'
    return redacted


def _parse_tool_output(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        return dict(raw_output)
    if not isinstance(raw_output, str):
        return {'raw': str(raw_output)}
    stripped = raw_output.strip()
    if not stripped:
        return {'raw': ''}
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return {'raw': stripped}
    return dict(parsed) if isinstance(parsed, dict) else {'raw': stripped}


def _tool_result_digest_for_retry(output: list[dict], *, max_items: int = 8) -> str:
    call_names: dict[str, str] = {}
    rows: list[str] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type == 'function_call':
            call_id = str(item.get('call_id') or item.get('id') or '')
            if call_id:
                call_names[call_id] = str(item.get('name') or 'tool')
            continue
        if item_type != 'function_call_output':
            continue
        parsed = _parse_tool_output(item.get('output'))
        status = str(parsed.get('status') or '').strip()
        exit_code = parsed.get('exit_code')
        stdout = _redact_report_text(str(parsed.get('stdout') or ''))
        stderr = _redact_report_text(str(parsed.get('stderr') or ''))
        raw = _redact_report_text(str(parsed.get('raw') or ''))
        body = stdout or stderr or raw
        if not body:
            continue
        call_id = str(item.get('call_id') or '')
        name = call_names.get(call_id) or 'tool'
        header = f'### Tool: {name}'
        meta = []
        if status:
            meta.append(f'status={status}')
        if exit_code is not None:
            meta.append(f'exit_code={exit_code}')
        if meta:
            header += f' ({", ".join(meta)})'
        rows.append(f'{header}\n```text\n{body}\n```')
    return '\n\n'.join(rows[-max_items:])


def _summary_text_for_retry(output: list[dict]) -> str:
    raw_text = _final_text_from_output(output)
    matches = list(_SUMMARY_BLOCK_RE.finditer(raw_text or ''))
    if not matches:
        return ''
    return ' '.join(matches[-1].group(1).split())


def _final_report_retry_input(output: list[dict]) -> str:
    digest = _tool_result_digest_for_retry(output)
    summary = _summary_text_for_retry(output)
    raw_text = _final_text_from_output(output)
    language = '中文' if _contains_cjk(raw_text + '\n' + digest) else 'the user primary language'
    if language == '中文':
        return (
            '上一轮已经完成工具执行，但最终只输出了 `<summary>`，缺少用户可读的最终报告。\n'
            '请不要再调用任何外部工具；只根据下面已有工具结果，调用 `final_report` 提交完整的 `report_markdown`。\n'
            '报告必须使用中文撰写；命令、字段名、路径、错误原文、代码和产品专有名词可以保留原文。\n'
            '报告应简洁，包含：结论、验证对象、验证方法、关键证据、风险或未完成项。不要只写 summary。\n\n'
            f'上一轮 summary：{summary or "(无)"}\n\n'
            f'已有工具结果：\n{digest or "(无可用工具输出)"}'
        )
    return (
        'The previous run completed tool execution but ended with only a `<summary>` and no user-readable final report.\n'
        'Do not call any external tools. Based only on the tool results below, call `final_report` with a complete `report_markdown`.\n'
        'Use the user primary language. Keep commands, field names, paths, original errors, code, and product names unchanged.\n'
        'Keep the report concise and include: conclusion, validated object, validation method, key evidence, and risks or open items. Do not output only summary.\n\n'
        f'Previous summary: {summary or "(none)"}\n\n'
        f'Tool results:\n{digest or "(no usable tool output)"}'
    )


def _final_report_retry_agent(agent):
    tools = [tool for tool in getattr(agent, 'tools', []) if getattr(tool, 'name', '') == 'final_report']
    if not tools:
        return agent
    return agent.clone(
        tools=tools,
        tool_use_behavior={'stop_at_tool_names': ['final_report']},
    )


def _string_arg(arguments: dict[str, Any], *names: str) -> str:
    for name in names:
        value = arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return ''


def _autonomous_hitl_resolution(envelope: InterruptionEnvelope) -> tuple[bool, str, str]:
    """Return ``(approve, answer, policy)`` for a HITL interruption.

    ``ask_user`` is a request for information, so autonomous mode approves it
    with either the model-provided default action or a conservative instruction.
    Other approval-gated tools are not user input requests; autonomous mode
    rejects them with a safe instruction so the model can choose an alternative
    instead of pausing the frontend.
    """
    if envelope.tool_name == 'ask_user':
        default = _string_arg(
            envelope.arguments,
            'default_action',
            'defaultAction',
            'default_answer',
            'default',
        )
        if default:
            if _PAUSE_LIKE_DEFAULT_ACTION_RE.search(default):
                return True, _AUTONOMOUS_ASK_USER_ANSWER, 'pause_like_default_rejected'
            return True, default, 'default_action'
        return True, _AUTONOMOUS_ASK_USER_ANSWER, 'conservative_default'
    return False, _AUTONOMOUS_REJECT_ANSWER, 'reject_non_input_approval'


def _autonomous_blocked_report(envelope: InterruptionEnvelope) -> str:
    question = _string_arg(envelope.arguments, 'question') or '(未提供问题)'
    return (
        '## 阻塞\n\n'
        '当前请求未开启用户打断，agent 多次请求用户输入后仍无法安全自主推进。\n\n'
        f'- 需要确认的问题：{question}\n'
        '- 当前处理：未继续执行需要用户确认的动作\n'
        '- 下一步：重新发起请求并设置 `allow_hitl=true`，或在用户消息中直接补充所需信息'
    )


def _append_hitl_auto_audit(
    audit_store: AuditStore | None,
    *,
    audit_log_id: str,
    ctx: RunContext,
    envelope: InterruptionEnvelope,
    answer: str,
    approve: bool,
    policy: str,
) -> None:
    if audit_store is None:
        return
    audit_store.append(AuditEvent(
        audit_log_id=audit_log_id,
        run_id=ctx.run_id,
        session_id=ctx.session_id,
        response_id=ctx.response_id,
        category=CATEGORY_HITL_AUTO_CONTINUE,
        payload={
            'call_id': envelope.call_id,
            'tool_name': envelope.tool_name,
            'question': envelope.arguments.get('question'),
            'risk': envelope.arguments.get('risk'),
            'reason': envelope.arguments.get('reason'),
            'default_action': envelope.arguments.get('default_action'),
            'approve': approve,
            'answer': answer,
            'policy': policy,
            'mode': 'autonomous',
        },
    ))


def _stream_text_chunks(text: str, *, max_chars: int = 96) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        split = end
        if end < len(text):
            window = text[start:end]
            for sep in ('\n\n', '\n', '。', '. ', '；', '; ', '，', ', '):
                pos = window.rfind(sep)
                if pos >= max(16, len(window) // 2):
                    split = start + pos + len(sep)
                    break
        chunks.append(text[start:split])
        start = split
    return chunks


def _finalize_open_message(
    state: ResponsesStreamState,
    *,
    response_id: str,
) -> list[dict]:
    chunks = event_bridge._complete_open_message_as_process_reasoning(
        state,
        response_id=response_id,
    )
    chunks.extend(event_bridge._close_step_if_open(state, response_id=response_id))
    return chunks


def _apply_final_report_contract(
    state: ResponsesStreamState,
    *,
    response_id: str,
    report_text: str,
) -> list[dict]:
    report_text = (report_text or '').strip()
    if not report_text:
        return []

    chunks = _finalize_open_message(state, response_id=response_id)
    for index, item in enumerate(state.output):
        if not _is_final_report_message(item):
            continue
        current_text = _message_output_text(item)
        if current_text.strip() == report_text and item.get('status') == 'completed':
            return chunks
        if report_text.startswith(current_text):
            for piece in _stream_text_chunks(report_text[len(current_text):]):
                chunks.append({
                    'type': 'response.output_text.delta',
                    'response_id': response_id,
                    'delta': piece,
                    'output_index': index,
                    'content_index': 0,
                })
            completed = {
                **item,
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': report_text}],
            }
            state.output[index] = completed
            chunks.extend([
                {
                    'type': 'response.output_text.done',
                    'response_id': response_id,
                    'text': report_text,
                    'output_index': index,
                    'content_index': 0,
                },
                {
                    'type': 'response.output_item.done',
                    'response_id': response_id,
                    'output_index': index,
                    'item': dict(completed),
                },
            ])
            return chunks

    idx = len(state.output)
    msg_id = f'msg_final_{response_id}'
    in_progress = {
        'id': msg_id,
        'type': 'message',
        'status': 'in_progress',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': ''}],
        'metadata': _pairag_metadata(**{_FINAL_REPORT_FLAG: True}),
    }
    completed = {
        **in_progress,
        'status': 'completed',
        'content': [{'type': 'output_text', 'text': report_text}],
    }
    state.output.append(in_progress)
    chunks.extend([
        {
            'type': 'response.output_item.added',
            'response_id': response_id,
            'output_index': idx,
            'item': dict(in_progress),
        },
    ])
    for piece in _stream_text_chunks(report_text):
        chunks.append({
            'type': 'response.output_text.delta',
            'response_id': response_id,
            'delta': piece,
            'output_index': idx,
            'content_index': 0,
        })
    chunks.extend([
        {
            'type': 'response.output_text.done',
            'response_id': response_id,
            'text': report_text,
            'output_index': idx,
            'content_index': 0,
        },
    ])
    state.output[idx] = completed
    chunks.append({
        'type': 'response.output_item.done',
        'response_id': response_id,
        'output_index': idx,
        'item': dict(completed),
    })
    return chunks


async def stream_responses_run(
    *,
    state_store: RunStateStore,
    audit_store: AuditStore | None,
    session_id: str | None,
    user_id: str,
    model: str,
    cwd: str,
    tools: list,
    input_text: str | None = None,
    input_items: list[dict[str, Any]] | None = None,
    previous_response_id: str | None = None,
    resume: ResumePayload | None = None,
    instructions_override: str | None = None,
    max_turns: int = 40,
    extras: dict[str, Any] | None = None,
    allow_hitl: bool = False,
) -> AsyncIterator[StreamFrame]:
    """Stream a run on the Responses-API wire.

    Normal new runs pass ``input_text`` or full-history ``input_items``.
    HITL resumes pass ``previous_response_id`` + ``resume``; ordinary
    ``previous_response_id`` continuation is resolved by the HTTP layer into
    ``input_items`` before this function is called.

    Yields :class:`StreamFrame` instances. The caller's job is to convert
    each ``frame.chunk`` into an SSE line and stop when ``frame.terminal``.
    """
    if previous_response_id is None and input_items is None and not input_text:
        raise ValueError('either input_text/input_items or previous_response_id+resume must be set')

    if previous_response_id:
        row = state_store.get(previous_response_id, user_id=user_id)
        if row is None:
            raise LookupError(f'run_state not found: {previous_response_id}')
        if row['status'] != RUN_STATE_REQUIRES_ACTION:
            raise LookupError(f'run not resumable: status={row["status"]}')
        if resume is None:
            raise ValueError('resume payload required when previous_response_id is set')
        async for frame in _resume_run(
            row=row,
            resume=resume,
            tools=tools,
            user_id=user_id,
            cwd=cwd,
            model=row['model'],
            audit_store=audit_store,
            state_store=state_store,
            instructions_override=instructions_override,
            max_turns=max_turns,
            extras=extras,
            allow_hitl=allow_hitl,
        ):
            yield frame
        return

    runner_input = input_items if input_items is not None else input_text
    sid = session_id or _mint_session_id()
    run_id = _mint_run_id()
    response_id = _mint_response_id()
    audit_log_id = _mint_audit_id()
    # Per-request Model: rotates the LLM credential via provider_pool so
    # concurrent runs spread across keys (see runtime_setup docstring).
    model_instance = acquire_request_model(model)
    agent = build_agent(model=model_instance, tools=tools, user_id=user_id, instructions_override=instructions_override)
    ctx = RunContext(
        session_id=sid, run_id=run_id, response_id=response_id,
        user_id=user_id, cwd=cwd, allow_hitl=allow_hitl,
    )
    register_run_extras(run_id, extras or {})
    register_inflight_response(response_id, sid)
    try:
        streaming = Runner.run_streamed(agent, input=runner_input, context=ctx, max_turns=max_turns)
        async for frame in _drive_stream(
            streaming=streaming,
            agent=agent,
            ctx=ctx,
            state_store=state_store,
            audit_store=audit_store,
            audit_log_id=audit_log_id,
            model=model,
            max_turns=max_turns,
            allow_hitl=allow_hitl,
            original_input=(
                input_text if isinstance(input_text, str)
                else json.dumps(input_items or '', ensure_ascii=False, default=str)
            ),
        ):
            yield frame
    finally:
        unregister_run_extras(run_id)
        unregister_inflight_response(response_id)


async def _resume_run(
    *,
    row: dict,
    resume: ResumePayload,
    tools: list,
    user_id: str,
    cwd: str,
    model: str,
    audit_store: AuditStore | None,
    state_store: RunStateStore,
    instructions_override: str | None,
    max_turns: int,
    extras: dict[str, Any] | None = None,
    allow_hitl: bool = False,
) -> AsyncIterator[StreamFrame]:
    sid = row['session_id']
    run_id = row['run_id']
    response_id = row['response_id'] or row['id']
    audit_log_id = row['audit_log_id']
    # Resume runs also rotate keys — a HITL pause might span hours, and the
    # key chosen on first leg may no longer be optimal (or alive).
    model_instance = acquire_request_model(model)
    agent = build_agent(model=model_instance, tools=tools, user_id=user_id, instructions_override=instructions_override)
    ctx = RunContext(
        session_id=sid, run_id=run_id, response_id=response_id,
        user_id=user_id, cwd=cwd, pending_human_answer=resume.answer,
        allow_hitl=allow_hitl,
    )
    state = await RunState.from_string(agent, row['run_state_blob'], context_override=ctx)
    matched = None
    for item in state.get_interruptions():
        call_id = getattr(item, 'call_id', None) or getattr(item, 'id', None)
        if str(call_id) == resume.call_id:
            matched = item
            break
    if matched is None:
        raise LookupError(f'no interruption matches call_id={resume.call_id!r}')
    register_run_extras(run_id, extras or {})
    register_inflight_response(response_id, sid)
    if resume.approve:
        state.approve(matched)
    else:
        state.reject(matched, rejection_message=resume.answer)
    if audit_store is not None:
        audit_store.append(AuditEvent(
            audit_log_id=audit_log_id, run_id=run_id, session_id=sid,
            response_id=response_id, category=CATEGORY_HITL_RESUME,
            payload={'call_id': resume.call_id, 'answer': resume.answer, 'approve': resume.approve},
        ))
    try:
        streaming = Runner.run_streamed(agent, input=state, max_turns=max_turns)
        async for frame in _drive_stream(
            streaming=streaming,
            agent=agent,
            ctx=ctx,
            state_store=state_store,
            audit_store=audit_store,
            audit_log_id=audit_log_id,
            model=model,
            max_turns=max_turns,
            allow_hitl=allow_hitl,
            original_input=resume.answer,
        ):
            yield frame
    finally:
        unregister_run_extras(run_id)
        unregister_inflight_response(response_id)


async def _drive_stream(
    *,
    streaming,
    agent,
    ctx: RunContext,
    state_store: RunStateStore,
    audit_store: AuditStore | None,
    audit_log_id: str,
    model: str,
    max_turns: int,
    allow_hitl: bool,
    original_input: str = '',
) -> AsyncIterator[StreamFrame]:
    """Iterate ``streaming.stream_events()`` and yield wire chunks /
    terminal frame. Caller is responsible for surrounding ``response.created``
    + ``response.completed`` envelopes when needed (kept in HTTP layer to
    preserve Responses-API event sequence numbers).
    """
    bridge_state = ResponsesStreamState()
    current_streaming = streaming
    final_state = None

    while True:
        try:
            async for sdk_event in current_streaming.stream_events():
                chunks = event_bridge.to_responses_chunk(sdk_event, bridge_state, response_id=ctx.response_id)
                for chunk in chunks:
                    yield StreamFrame(chunk=chunk)
                if audit_store is not None:
                    row = event_bridge.to_audit(sdk_event)
                    if row is not None:
                        if row['category'] == CATEGORY_HITL_PAUSE and not allow_hitl:
                            continue
                        audit_store.append(AuditEvent(
                            audit_log_id=audit_log_id, run_id=ctx.run_id,
                            session_id=ctx.session_id, response_id=ctx.response_id,
                            category=row['category'], payload=row['payload'],
                        ))
        except Exception as e:
            if audit_store is not None:
                audit_store.append(AuditEvent(
                    audit_log_id=audit_log_id, run_id=ctx.run_id,
                    session_id=ctx.session_id, response_id=ctx.response_id,
                    category=CATEGORY_RUN_FAILED, payload={'error': str(e)},
                ))
            state_store.upsert(
                id=ctx.response_id, session_id=ctx.session_id, run_id=ctx.run_id,
                response_id=ctx.response_id, user_id=ctx.user_id, model=model,
                status=RUN_STATE_FAILED, run_state_blob='', pending_interruption_json=None,
                last_event_id=None, audit_log_id=audit_log_id,
            )
            yield StreamFrame(terminal=True, response_object={
                'id': ctx.response_id, 'object': 'response', 'status': 'failed',
                'model': model, 'output': bridge_state.output,
                'error': {'message': str(e)},
            })
            return

        final_state = current_streaming.to_state()
        interruptions = list(final_state.get_interruptions() or [])
        if not interruptions:
            break

        envelope = interruption_from_sdk_item(interruptions[0])
        if not allow_hitl:
            if ctx.auto_hitl_count >= _AUTO_HITL_MAX_CONTINUES:
                for chunk in _apply_final_report_contract(
                    bridge_state,
                    response_id=ctx.response_id,
                    report_text=_autonomous_blocked_report(envelope),
                ):
                    yield StreamFrame(chunk=chunk)
                break
            approve, answer, policy = _autonomous_hitl_resolution(envelope)
            ctx.pending_human_answer = answer
            ctx.auto_hitl_count += 1
            if approve:
                final_state.approve(interruptions[0])
            else:
                final_state.reject(interruptions[0], rejection_message=answer)
            _append_hitl_auto_audit(
                audit_store,
                audit_log_id=audit_log_id,
                ctx=ctx,
                envelope=envelope,
                answer=answer,
                approve=approve,
                policy=policy,
            )
            current_streaming = Runner.run_streamed(agent, input=final_state, max_turns=max_turns)
            continue

        blob = final_state.to_string()
        pending_json = json.dumps(
            [
                {
                    'call_id': interruption_from_sdk_item(item).call_id,
                    'tool_name': interruption_from_sdk_item(item).tool_name,
                    'arguments': interruption_from_sdk_item(item).arguments,
                }
                for item in interruptions
            ],
            ensure_ascii=False,
        )
        state_store.upsert(
            id=ctx.response_id, session_id=ctx.session_id, run_id=ctx.run_id,
            response_id=ctx.response_id, user_id=ctx.user_id, model=model,
            status=RUN_STATE_REQUIRES_ACTION, run_state_blob=blob,
            pending_interruption_json=pending_json, last_event_id=None,
            audit_log_id=audit_log_id,
        )
        action_item = envelope.serialize(wire='responses')
        # Surface the action item as one final output_item.added so clients
        # using only the streaming wire see what's pending without a follow-up
        # GET. Then yield a terminal "requires_action" envelope.
        # Dedup: the same function_call may already be in bridge_state.output
        # from the upstream raw stream (output_item.added(function_call) +
        # arguments.delta/done). If so, skip both the append and the
        # synthesized output_item.added — the client already has it.
        existing_idx = None
        for i, entry in enumerate(bridge_state.output):
            if entry.get('type') == 'function_call' and entry.get('call_id') == envelope.call_id:
                existing_idx = i
                break
        if existing_idx is None:
            action_index = len(bridge_state.output)
            bridge_state.output.append(action_item)
            yield StreamFrame(chunk={
                'type': 'response.output_item.added',
                'response_id': ctx.response_id,
                'output_index': action_index,
                'item': dict(action_item),
            })
        response_obj = {
            'id': ctx.response_id, 'object': 'response',
            'status': 'requires_action', 'model': model,
            'output': bridge_state.output,
            'required_action': {
                'type': 'submit_tool_outputs',
                'submit_tool_outputs': {'tool_calls': [
                    {
                        'id': envelope.call_id,
                        'type': 'function',
                        'function': {
                            'name': envelope.tool_name,
                            'arguments': json.dumps(envelope.arguments, ensure_ascii=False),
                        },
                    }
                    for envelope in (interruption_from_sdk_item(i) for i in interruptions)
                ]},
            },
        }
        yield StreamFrame(terminal=True, response_object=response_obj, interruption=envelope)
        return

    if _needs_tool_intent_retry(bridge_state.output):
        retry_streaming = Runner.run_streamed(
            agent,
            input=_tool_intent_retry_input(bridge_state.output, original_input),
            context=ctx,
            max_turns=max(1, min(max_turns, 4)),
        )
        async for sdk_event in retry_streaming.stream_events():
            chunks = event_bridge.to_responses_chunk(sdk_event, bridge_state, response_id=ctx.response_id)
            for chunk in chunks:
                yield StreamFrame(chunk=chunk)
            if audit_store is not None:
                row = event_bridge.to_audit(sdk_event)
                if row is not None:
                    audit_store.append(AuditEvent(
                        audit_log_id=audit_log_id, run_id=ctx.run_id,
                        session_id=ctx.session_id, response_id=ctx.response_id,
                        category=row['category'], payload=row['payload'],
                    ))
        final_state = retry_streaming.to_state()
        current_streaming = retry_streaming

    if not _final_report_from_extras(ctx) and _needs_final_report_retry(bridge_state.output):
        retry_streaming = Runner.run_streamed(
            _final_report_retry_agent(agent),
            input=_final_report_retry_input(bridge_state.output),
            context=ctx,
            max_turns=2,
        )
        async for sdk_event in retry_streaming.stream_events():
            chunks = event_bridge.to_responses_chunk(sdk_event, bridge_state, response_id=ctx.response_id)
            for chunk in chunks:
                yield StreamFrame(chunk=chunk)
            if audit_store is not None:
                row = event_bridge.to_audit(sdk_event)
                if row is not None:
                    audit_store.append(AuditEvent(
                        audit_log_id=audit_log_id, run_id=ctx.run_id,
                        session_id=ctx.session_id, response_id=ctx.response_id,
                        category=row['category'], payload=row['payload'],
                    ))
        final_state = retry_streaming.to_state()

    final_report_text = (
        _final_report_from_extras(ctx)
        or _final_report_from_text_wrapper(bridge_state.output)
    )
    if final_report_text:
        for chunk in _apply_final_report_contract(
            bridge_state,
            response_id=ctx.response_id,
            report_text=final_report_text,
        ):
            yield StreamFrame(chunk=chunk)

    _finalize_output_for_responses(bridge_state)

    final_text = (
        final_report_text
        or _final_text_from_output(bridge_state.output)
        or ''.join(bridge_state.accumulated_text)
    )
    usage = _extract_usage(current_streaming)
    if audit_store is not None:
        row = event_bridge.run_complete_audit(final_text=final_text, usage=usage)
        audit_store.append(AuditEvent(
            audit_log_id=audit_log_id, run_id=ctx.run_id,
            session_id=ctx.session_id, response_id=ctx.response_id,
            category=row['category'], payload=row['payload'],
        ))
    state_store.upsert(
        id=ctx.response_id, session_id=ctx.session_id, run_id=ctx.run_id,
        response_id=ctx.response_id, user_id=ctx.user_id, model=model,
        status=RUN_STATE_COMPLETED, run_state_blob=final_state.to_string(),
        pending_interruption_json=None, last_event_id=None,
        audit_log_id=audit_log_id,
    )
    response_obj = {
        'id': ctx.response_id, 'object': 'response', 'status': 'completed',
        'model': model, 'output': bridge_state.output,
        'output_text': final_text,
        'usage': usage,
    }
    yield StreamFrame(terminal=True, response_object=response_obj)


def _extract_usage(streaming) -> dict | None:
    """Pull token usage off the streaming run for ``response.completed``.

    SDK accumulates usage on ``RunContextWrapper.usage`` (a ``Usage`` dataclass)
    via ``usage_delta`` after each model turn. ``RunResultStreaming`` exposes
    the wrapper directly as ``context_wrapper`` (inherited from
    ``RunResultBase``). Earlier code read ``streaming.to_state().usage`` —
    ``RunState`` has no such field, so usage was always ``None``.

    We project to the OpenAI Responses ``response.usage`` shape (drop
    SDK-specific ``requests`` and ``request_usage_entries``) so business
    callers see the same fields they'd get from the upstream API.
    """
    try:
        wrapper = getattr(streaming, 'context_wrapper', None)
        usage = getattr(wrapper, 'usage', None) if wrapper is not None else None
        if usage is None:
            return None
        def _details(obj: Any, default_key: str) -> dict:
            if obj is None:
                return {default_key: 0}
            if hasattr(obj, 'model_dump'):
                return obj.model_dump(exclude_none=False)
            if isinstance(obj, dict):
                return obj
            return {default_key: getattr(obj, default_key, 0) or 0}
        input_tokens = int(getattr(usage, 'input_tokens', 0) or 0)
        output_tokens = int(getattr(usage, 'output_tokens', 0) or 0)
        total_tokens = int(getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens))
        if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
            return None
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_tokens_details': _details(getattr(usage, 'input_tokens_details', None), 'cached_tokens'),
            'output_tokens_details': _details(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens'),
        }
    except Exception:
        return None
