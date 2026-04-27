"""ACP-style agent event helpers shared by HTTP and ACP frontends."""
import json
import re
from typing import Any


THINKING_RE = re.compile(r'<thinking>\s*(.*?)\s*</thinking>', re.DOTALL)
SUMMARY_RE = re.compile(r'<summary>\s*(.*?)\s*</summary>', re.DOTALL)
INTERNAL_TOOL_NAMES = {
    'update_working_checkpoint',
    'start_long_term_update',
}


def text_content(text: str) -> dict:
    return {'type': 'text', 'text': text or ''}


def agent_message_chunk(text: str) -> dict:
    return {
        'sessionUpdate': 'agent_message_chunk',
        'content': text_content(text),
    }


def thought(content: str, title: str = 'Thinking') -> dict:
    return {
        'sessionUpdate': 'thought',
        'title': title,
        'content': text_content(content),
    }


def thought_start(thought_id: str, title: str = 'Agent step') -> dict:
    return {
        'sessionUpdate': 'thought_start',
        'thoughtId': thought_id,
        'title': title,
        'status': 'in_progress',
    }


def thought_delta(thought_id: str, content: str, replace: bool = False) -> dict:
    event = {
        'sessionUpdate': 'thought_delta',
        'thoughtId': thought_id,
        'content': text_content(content),
    }
    if replace:
        event['replace'] = True
    return event


def thought_done(
    thought_id: str,
    status: str = 'completed',
    hidden: bool = False,
    content: str | None = None,
) -> dict:
    event = {
        'sessionUpdate': 'thought_done',
        'thoughtId': thought_id,
        'status': status,
    }
    if hidden:
        event['hidden'] = True
    if content is not None:
        event['content'] = text_content(content)
    return event


def ask_user(question: str, candidates: list[str] | None = None) -> dict:
    return {
        'sessionUpdate': 'ask_user',
        'question': question,
        'candidates': candidates or [],
    }


def done(stop_reason: str = 'end_turn') -> dict:
    return {
        'sessionUpdate': 'done',
        'stopReason': stop_reason,
    }


def tool_kind(name: str) -> str:
    if name in ('file_read',):
        return 'read'
    if name in ('file_write', 'file_patch'):
        return 'edit'
    if name in ('code_run',):
        return 'execute'
    if name in ('ask_user',):
        return 'ask'
    if name in ('update_working_checkpoint', 'start_long_term_update', 'use_skill'):
        return 'think'
    return 'other'


def tool_title(name: str, args: dict[str, Any] | None = None) -> str:
    args = args or {}
    if name in ('file_read', 'file_write', 'file_patch') and args.get('path'):
        return f'{name}: {args["path"]}'
    if name == 'code_run':
        return f'code_run: {args.get("type", "python")}'
    if name == 'ask_user':
        return 'Ask user'
    if name == 'use_skill' and args.get('skill'):
        return f'use_skill: {args["skill"]}'
    return name


def is_internal_tool(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def tool_call(tool_call_id: str, name: str, args: dict[str, Any] | None = None) -> dict:
    clean_args = {k: v for k, v in (args or {}).items() if not str(k).startswith('_')}
    event = {
        'sessionUpdate': 'tool_call',
        'toolCallId': tool_call_id,
        'title': tool_title(name, clean_args),
        'name': name,
        'kind': tool_kind(name),
        'status': 'pending',
        'input': clean_args,
    }
    if is_internal_tool(name):
        event['hidden'] = True
    return event


def tool_call_delta(
    tool_call_id: str,
    index: int,
    name: str = '',
    name_delta: str = '',
    arguments_delta: str = '',
    arguments_text: str = '',
) -> dict:
    event = {
        'sessionUpdate': 'tool_call_delta',
        'toolCallId': tool_call_id,
        'index': index,
        'name': name,
        'title': tool_title(name) if name else 'Tool call',
        'kind': tool_kind(name) if name else 'other',
        'status': 'in_progress',
        'argumentsText': arguments_text or '',
    }
    if name_delta:
        event['nameDelta'] = name_delta
    if arguments_delta:
        event['argumentsDelta'] = arguments_delta
    if is_internal_tool(name):
        event['hidden'] = True
    return event


def tool_call_update(tool_call_id: str, status: str, content: str = '', data: Any = None) -> dict:
    update = {
        'sessionUpdate': 'tool_call_update',
        'toolCallId': tool_call_id,
        'status': status,
    }
    if content:
        update['content'] = text_content(content)
    if data is not None:
        update['data'] = data
    return update


def split_model_content(content: str) -> tuple[list[str], str]:
    thoughts = [m.group(1).strip() for m in THINKING_RE.finditer(content or '') if m.group(1).strip()]
    cleaned = THINKING_RE.sub('', content or '')
    cleaned = SUMMARY_RE.sub('', cleaned).strip()
    return thoughts, cleaned


def stream_model_process_content(content: str) -> str:
    """Return displayable model-process text from a partial model stream.

    This hides protocol tags as soon as they are recognizable, includes open
    <thinking> blocks before they close, and suppresses <summary> content.
    """
    text = content or ''
    last_lt = text.rfind('<')
    last_gt = text.rfind('>')
    if last_lt > last_gt:
        text = text[:last_lt]

    pieces = []
    pos = 0
    while pos < len(text):
        thinking_at = text.find('<thinking>', pos)
        summary_at = text.find('<summary>', pos)
        starts = [(idx, name) for idx, name in (
            (thinking_at, 'thinking'),
            (summary_at, 'summary'),
        ) if idx != -1]
        if not starts:
            pieces.append(text[pos:])
            break

        start, name = min(starts, key=lambda item: item[0])
        pieces.append(text[pos:start])
        open_tag = f'<{name}>'
        close_tag = f'</{name}>'
        inner_start = start + len(open_tag)
        end = text.find(close_tag, inner_start)

        if name == 'thinking':
            if end == -1:
                pieces.append(text[inner_start:])
                break
            pieces.append(text[inner_start:end])
            pos = end + len(close_tag)
            continue

        if end == -1:
            break
        pos = end + len(close_tag)

    return ''.join(pieces).strip()


def stringify(data: Any) -> str:
    if data is None:
        return ''
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, default=str)
    return str(data)


def stop_reason(exit_reason: dict | None) -> str:
    result = (exit_reason or {}).get('result')
    if result == 'MAX_TURNS_EXCEEDED':
        return 'max_turn_requests'
    if result == 'INTERRUPTED':
        return 'cancelled'
    if result == 'ERROR':
        return 'refusal'
    return 'end_turn'
