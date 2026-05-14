"""精简版执行循环。
- 每轮：build new_messages → client.chat → 解析 tool_calls
- 工具结果按 OpenAI 原生 {'role':'tool', 'tool_call_id':...} 消息回填
- Web/ACP 通过结构化 AgentEvent 输出
- 退出条件：should_exit / 无 next_prompt / 达到 max_turns
"""
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

from agent_events import (
    agent_message_chunk,
    done,
    model_summary_content,
    split_model_content,
    stop_reason,
    stream_model_process_content,
    stringify,
    thought,
    thought_delta,
    thought_done,
    thought_start,
    tool_call,
    tool_call_delta,
    tool_call_update,
)
from llm_client import ToolCallDelta

SUMMARY_ONLY_FINAL_RETRY_PROMPT = (
    '上一轮你只输出了内部思考/规划标签和 <summary>...</summary>，没有用户可见的最终回答正文。\n'
    '<summary> 只是内部历史摘要，前端不会把它当作最终回答展示。\n'
    '请先判断任务是否已经真正完成：\n'
    '- 如果已经完成，立即输出用户可见的最终报告正文；\n'
    '- 如果尚未完成，继续调用必要工具推进任务，不要只做计划或复述。\n'
    '不要只输出 <summary>。如果需要保留摘要，只能放在报告正文之后。\n'
    '复杂诊断类最终报告必须包含：诊断结论、日志证据、配置证据、因果链路、证据边界。'
)

TOOL_INTENT_WITHOUT_CALL_RETRY_PROMPT = (
    '上一轮你在文本里声称要调用工具或激活 skill（例如「我将向用户询问」「调用 ask_user」'
    '「启动/激活 X 技能」「use_skill …」「我将暂停流程」或类似表述），但本轮实际并没有发起'
    '任何 tool_call，前端因此什么也没收到。\n'
    '请二选一：\n'
    '- 如果确实需要调用工具或激活 skill，立即真正发起对应 tool_call（例如 ask_user / use_skill '
    '必须以工具调用形式发出，不能只在文本或 <summary> 里描述）；\n'
    '- 如果不需要再调用工具，直接输出用户可见的最终回答正文，不要再用「我将…」之类的预告口吻。'
)

TOOL_INTENT_WITHOUT_CALL_RE = re.compile(
    r'(?:'
    r'调用\s*(?:ask_user|工具|tool)'
    r'|(?:向|跟|与)\s*用户\s*(?:发起|进行|做出)?\s*(?:明确)?\s*询问'
    r'|我(?:将|会|准备|打算|要)\s*(?:暂停|向用户|对用户|询问用户|发起询问|调用|使用)'
    r'|因此[，,]\s*我(?:将|会|要|准备|打算)'
    r'|(?:启动|激活|调用|使用|开启|进入)\s*[\w\-/. ]{0,40}?\s*(?:技能|skill)'
    r'|use_skill'
    r"|I\s+(?:will|am\s+going\s+to|need\s+to|have\s+to)\s+(?:call|ask|invoke|use)"
    r')',
    re.IGNORECASE,
)

MAX_TURNS_FALLBACK_PROMPT = (
    '已达到本次 agent 最大执行轮次，不能再调用工具。\n'
    '请基于目前已经完成的工具结果和对话上下文，输出用户可见的部分完成报告。\n'
    '报告必须结构化说明：\n'
    '1. 已完成的检查或修改；\n'
    '2. 已确认的证据和结论；\n'
    '3. 尚未完成的原因或阻塞点；\n'
    '4. 建议的下一步。\n'
    '不要输出内部思考标签；不要只输出 <summary>。'
)

TOOL_RESULT_MAX_CHARS = 100000
FILE_READ_RESULT_MAX_CHARS = 12000
TURN_TOOL_RESULT_BUDGET_CHARS = 200000
TOOL_RESULT_PREVIEW_CHARS = 12000
PERSISTED_OUTPUT_TAG = '<persisted-output>'
ASSISTANT_OUTPUT_MAX_CHARS = 24000
ASSISTANT_FILE_CONTENT_MAX_CHARS = 8000
ASSISTANT_STREAM_PREVIEW_CHARS = 12000
ARCHIVE_TEXT_MAX_CHARS = 30000
ARCHIVE_LIST_MAX_ITEMS = 200
ARCHIVE_DICT_MAX_ITEMS = 200

SECRET_PATTERNS = (
    re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.DOTALL | re.IGNORECASE),
    re.compile(r'\bLTAI[A-Za-z0-9]{12,}\b'),
    re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
    re.compile(
        r'(?i)(["\']?[A-Za-z0-9_.-]{0,80}(?:api[_-]?key|secret|token|password|passwd|'
        r'access[_-]?key|secret[_-]?key|auth[_-]?secret)[A-Za-z0-9_.-]{0,80}["\']?\s*[:=]\s*)'
        r'(["\']?)([^"\'\s,}\]]{4,})(["\']?)'
    ),
)
SECRET_KEY_RE = re.compile(
    r'(?i)(api[_-]?key|secret|token|password|passwd|access[_-]?key|secret[_-]?key|auth[_-]?secret)'
)


@dataclass
class StepOutcome:
    data: Any
    next_prompt: Optional[str] = None
    should_exit: bool = False


class BaseHandler:
    """子类实现 do_<tool_name>(args, response) -> StepOutcome。"""

    def turn_end_callback(self, response, tool_calls, tool_results, turn,
                          next_prompt, exit_reason):
        return next_prompt

    def dispatch(self, tool_name, args, response, index=0):
        method = getattr(self, f'do_{tool_name}', None)
        if method is None:
            return StepOutcome(None,
                               next_prompt=f'未知工具 {tool_name}',
                               should_exit=False)
        args['_index'] = index
        return method(args, response)

    def persist_tool_result(self, content, tool_call_id, subdir='tool_results'):
        return None


def _head_tail_text(text, max_chars=TOOL_RESULT_PREVIEW_CHARS):
    if len(text) <= max_chars:
        return text
    notice = f'\n\n[OUTPUT PREVIEW TRUNCATED: {len(text) - max_chars} chars omitted from {len(text)} total]\n\n'
    keep_budget = max(max_chars - len(notice), 0)
    if keep_budget <= 1:
        return notice.strip()
    head_chars = max(1, int(keep_budget * 0.4))
    tail_chars = max(1, keep_budget - head_chars)
    return f'{text[:head_chars]}{notice}{text[-tail_chars:]}'


def _stream_preview_text(text, max_chars=ASSISTANT_STREAM_PREVIEW_CHARS):
    text = redact_sensitive_text(text or '')
    if len(text) <= max_chars:
        return text
    notice = (
        f'\n\n[ASSISTANT OUTPUT STREAM TRUNCATED after {max_chars} chars; '
        'complete content will be saved to a workspace file if this response is retained]\n'
    )
    keep_budget = max(max_chars - len(notice), 0)
    return f'{text[:keep_budget]}{notice}'


def _persisted_output_message(content, path, reason):
    preview = _head_tail_text(redact_sensitive_text(content))
    return (
        f'{PERSISTED_OUTPUT_TAG}\n'
        f'reason: {reason}\n'
        f'chars: {len(content)}\n'
        f'path: {path}\n'
        'instruction: This message contains only a preview. Use the path above to inspect '
        'the complete output before parsing structured data or making evidence-sensitive conclusions.\n'
        f'</persisted-output>\n\n'
        f'<preview>\n{preview}\n</preview>'
    )


def _persist_tool_result_if_needed(handler, content, tool_name, tool_call_id, force=False, reason=None):
    if not isinstance(content, str):
        content = str(content)
    if PERSISTED_OUTPUT_TAG in content:
        return content
    max_chars = FILE_READ_RESULT_MAX_CHARS if tool_name == 'file_read' else TOOL_RESULT_MAX_CHARS
    if not force and len(content) <= max_chars:
        return content

    path = None
    try:
        path = handler.persist_tool_result(content, tool_call_id, subdir='tool_results')
    except Exception as e:
        path = None
        reason = f'{reason or "tool result too large"}; persist failed: {type(e).__name__}: {e}'
    if not path:
        return _head_tail_text(content)
    return _persisted_output_message(content, path, reason or 'tool result too large')


def _enforce_tool_result_budget(handler, tool_results, tool_result_meta):
    total = sum(len(msg.get('content') or '') for msg in tool_results)
    while total > TURN_TOOL_RESULT_BUDGET_CHARS:
        candidates = [
            (len(msg.get('content') or ''), index)
            for index, msg in enumerate(tool_results)
            if PERSISTED_OUTPUT_TAG not in (msg.get('content') or '')
        ]
        if not candidates:
            break
        _, index = max(candidates)
        msg = tool_results[index]
        meta = tool_result_meta[index]
        original = msg.get('content') or ''
        msg['content'] = _persist_tool_result_if_needed(
            handler,
            original,
            meta.get('tool_name', ''),
            meta.get('tool_call_id', ''),
            force=True,
            reason='turn tool-result budget exceeded',
        )
        new_total = sum(len(item.get('content') or '') for item in tool_results)
        if new_total >= total:
            break
        total = new_total


def _tool_event_data(outcome_data, result_text):
    if PERSISTED_OUTPUT_TAG not in (result_text or ''):
        return outcome_data
    status = outcome_data.get('status') if isinstance(outcome_data, dict) else 'success'
    return {
        'status': status or 'success',
        'result_persisted': True,
        'result_preview': result_text,
    }


def _model_process_content(content):
    thoughts, cleaned = split_model_content(content or '')
    parts = thoughts[:]
    if cleaned:
        parts.append(cleaned)
    return '\n\n'.join(parts).strip()


def _model_final_answer(content):
    _, cleaned = split_model_content(content or '')
    return cleaned


def _model_summary(content):
    return model_summary_content(content or '')


def _assistant_persist_reason(content, stop_reason=''):
    text = content or ''
    if PERSISTED_OUTPUT_TAG in text:
        return ''
    if '<file_content' in text.lower() and len(text) > ASSISTANT_FILE_CONTENT_MAX_CHARS:
        return 'assistant file_content output too large'
    if len(text) > ASSISTANT_OUTPUT_MAX_CHARS:
        return 'assistant output too large'
    if stop_reason == 'length' and len(text) > ASSISTANT_STREAM_PREVIEW_CHARS:
        return 'assistant output hit model max_tokens'
    return ''


def _replace_last_assistant_content(client, old_content, new_content):
    history = getattr(client, 'history', None)
    if not isinstance(history, list):
        return
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            msg['content'] = new_content
            notify = getattr(client, '_notify_history_changed', None)
            if callable(notify):
                notify()
            return


def _persist_assistant_response_if_needed(handler, client, response, step_id, reason=None):
    content = response.content or ''
    reason = reason or _assistant_persist_reason(content, getattr(response, 'stop_reason', '') or '')
    if not reason:
        return response.content
    path = None
    if handler is not None:
        try:
            path = handler.persist_tool_result(content, step_id, subdir='assistant_outputs')
        except Exception as e:
            reason = f'{reason}; persist failed: {type(e).__name__}: {e}'
    if path:
        replacement = _persisted_output_message(content, path, reason)
    else:
        replacement = _head_tail_text(content)
    response.content = replacement
    _replace_last_assistant_content(client, content, replacement)
    return replacement


class _PlainStreamLimiter:
    def __init__(self, max_chars=ASSISTANT_STREAM_PREVIEW_CHARS):
        self.max_chars = max_chars
        self.count = 0
        self.truncated = False

    def filter(self, chunk):
        if not chunk or self.truncated:
            return ''
        remaining = self.max_chars - self.count
        if len(chunk) <= remaining:
            self.count += len(chunk)
            return redact_sensitive_text(chunk)
        self.truncated = True
        head = chunk[:max(remaining, 0)]
        return redact_sensitive_text(
            head
            + f'\n\n[ASSISTANT OUTPUT STREAM TRUNCATED after {self.max_chars} chars; '
            'complete content will be saved to a workspace file if this response is retained]\n'
        )


def redact_sensitive_text(text):
    redacted = '' if text is None else str(text)
    for pattern in SECRET_PATTERNS:
        def repl(match):
            if match.lastindex and match.lastindex >= 4:
                return f'{match.group(1)}{match.group(2)}[REDACTED]{match.group(4)}'
            return '[REDACTED]'
        redacted = pattern.sub(repl, redacted)
    return redacted


def sanitize_for_archive(value, max_text_chars=ARCHIVE_TEXT_MAX_CHARS):
    if isinstance(value, str):
        redacted = redact_sensitive_text(value)
        return _head_tail_text(redacted, max_text_chars)
    if isinstance(value, dict):
        items = list(value.items())
        sanitized = {}
        for key, val in items[:ARCHIVE_DICT_MAX_ITEMS]:
            if SECRET_KEY_RE.search(str(key)):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = sanitize_for_archive(val, max_text_chars=max_text_chars)
        if len(items) > ARCHIVE_DICT_MAX_ITEMS:
            sanitized['__archive_truncated__'] = f'{len(items) - ARCHIVE_DICT_MAX_ITEMS} dict entries omitted'
        return sanitized
    if isinstance(value, list):
        sanitized = [
            sanitize_for_archive(item, max_text_chars=max_text_chars)
            for item in value[:ARCHIVE_LIST_MAX_ITEMS]
        ]
        if len(value) > ARCHIVE_LIST_MAX_ITEMS:
            sanitized.append({'__archive_truncated__': f'{len(value) - ARCHIVE_LIST_MAX_ITEMS} list items omitted'})
        return sanitized
    return value


def _emit_model_response(client, system_prompt, new_messages, tools_schema, model_step_id,
                         title, handler=None, on_chunk=None, on_event=None):
    gen = client.chat(system=system_prompt, new_messages=new_messages, tools=tools_schema)
    model_raw_content = ''
    model_process_content = ''
    plain_limiter = _PlainStreamLimiter()
    if on_event:
        on_event(thought_start(model_step_id, title=title))
    try:
        while True:
            chunk = next(gen)
            if isinstance(chunk, ToolCallDelta):
                continue
            if on_event:
                model_raw_content += chunk
                next_process_content = _stream_preview_text(stream_model_process_content(model_raw_content))
                if next_process_content.startswith(model_process_content):
                    delta = next_process_content[len(model_process_content):]
                    replace = False
                else:
                    delta = next_process_content
                    replace = True
                if delta or replace:
                    on_event(thought_delta(model_step_id, delta, replace=replace))
                model_process_content = next_process_content
            elif on_chunk:
                display_chunk = plain_limiter.filter(chunk)
                if display_chunk:
                    on_chunk(display_chunk)
            else:
                display_chunk = plain_limiter.filter(chunk)
                if display_chunk:
                    sys.stdout.write(display_chunk)
                    sys.stdout.flush()
    except StopIteration as e:
        response = e.value
    if not on_event and not on_chunk:
        print()
    original_content = response.content
    _persist_assistant_response_if_needed(handler, client, response, model_step_id)
    if response.content != original_content:
        if on_chunk:
            on_chunk('\n\n' + response.content)
        elif not on_event:
            print(response.content)
    return response


def _run_max_turns_fallback(client, system_prompt, pending_messages, max_turns,
                            handler=None, on_chunk=None, on_event=None):
    fallback_messages = list(pending_messages or [])
    fallback_messages.append({'role': 'user', 'content': MAX_TURNS_FALLBACK_PROMPT})
    response = _emit_model_response(
        client,
        system_prompt,
        fallback_messages,
        [],
        f'model-{max_turns + 1}-fallback',
        'Final report',
        handler=handler,
        on_chunk=on_chunk,
        on_event=on_event,
    )
    process_content = _model_process_content(response.content)
    cleaned = _model_final_answer(response.content)
    summary = _model_summary(response.content)
    visible_reply = cleaned or ('' if summary else process_content) or summary
    if on_event:
        on_event(thought_done(
            f'model-{max_turns + 1}-fallback',
            hidden=True,
            content=_stream_preview_text(process_content),
        ))
        if visible_reply:
            on_event(agent_message_chunk(visible_reply))
    return response, visible_reply


def agent_runner_loop(client, system_prompt, user_input, handler, tools_schema,
                      max_turns=40, on_chunk=None, on_event=None):
    """主循环。第一轮用 user_input 文本启动；之后用 tool_results + next_prompt。"""
    handler.max_turns = max_turns
    new_messages = [{'role': 'user', 'content': user_input}]
    exit_reason = {}
    summary_only_retry_count = 0
    tool_intent_retry_used = False
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def _accumulate(resp):
        u = getattr(resp, 'usage', None) or {}
        total_usage['prompt_tokens'] += int(u.get('prompt_tokens', 0) or 0)
        total_usage['completion_tokens'] += int(u.get('completion_tokens', 0) or 0)
        total_usage['total_tokens'] += int(u.get('total_tokens', 0) or 0)

    for turn in range(1, max_turns + 1):
        handler.current_turn = turn
        # 1) 调用 LLM
        gen = client.chat(system=system_prompt, new_messages=new_messages, tools=tools_schema)
        model_step_id = f'model-{turn}'
        model_raw_content = ''
        model_process_content = ''
        plain_limiter = _PlainStreamLimiter()
        plain_stream_emitted = False
        if on_event:
            on_event(thought_start(model_step_id, title='Agent step'))
        try:
            while True:
                chunk = next(gen)
                if isinstance(chunk, ToolCallDelta):
                    if on_event:
                        on_event(tool_call_delta(
                            f'call_{turn}_{chunk.index}',
                            index=chunk.index,
                            name=chunk.name,
                            name_delta=chunk.name_delta,
                            arguments_delta=chunk.arguments_delta,
                            arguments_text=chunk.arguments_text,
                        ))
                    continue
                if on_event:
                    model_raw_content += chunk
                    next_process_content = _stream_preview_text(stream_model_process_content(model_raw_content))
                    if next_process_content.startswith(model_process_content):
                        delta = next_process_content[len(model_process_content):]
                        replace = False
                    else:
                        delta = next_process_content
                        replace = True
                    if delta or replace:
                        on_event(thought_delta(model_step_id, delta, replace=replace))
                    model_process_content = next_process_content
                elif on_chunk:
                    model_raw_content += chunk
                    next_process_content = _stream_preview_text(stream_model_process_content(model_raw_content))
                    if next_process_content.startswith(model_process_content):
                        display_chunk = next_process_content[len(model_process_content):]
                    else:
                        display_chunk = next_process_content
                    model_process_content = next_process_content
                    display_chunk = plain_limiter.filter(display_chunk)
                    if display_chunk:
                        on_chunk(display_chunk)
                        plain_stream_emitted = True
                else:
                    display_chunk = plain_limiter.filter(chunk)
                    if display_chunk:
                        sys.stdout.write(display_chunk)
                        sys.stdout.flush()
        except StopIteration as e:
            response = e.value
        _accumulate(response)
        if not on_event and not on_chunk:
            print()

        # 2) 解析 tool_calls
        tool_calls = [{'tool_name': tc.name, 'args': tc.input, 'id': tc.id}
                      for tc in response.tool_calls]
        if not tool_calls:
            original_content = response.content
            _persist_assistant_response_if_needed(handler, client, response, model_step_id)
            assistant_replaced = response.content != original_content
            process_content = _model_process_content(response.content)
            cleaned = _model_final_answer(response.content)
            summary = _model_summary(response.content)
            visible_reply = cleaned or ('' if summary else process_content)
            if on_event:
                on_event(thought_done(
                    model_step_id,
                    hidden=True,
                    content=process_content,
                ))

            if not visible_reply and summary and summary_only_retry_count < 2 and turn < max_turns:
                summary_only_retry_count += 1
                handler.turn_end_callback(response, [], [], turn, '', {})
                new_messages = [{'role': 'user', 'content': SUMMARY_ONLY_FINAL_RETRY_PROMPT}]
                continue

            if (
                not tool_intent_retry_used
                and turn < max_turns
                and TOOL_INTENT_WITHOUT_CALL_RE.search(response.content or '')
            ):
                tool_intent_retry_used = True
                handler.turn_end_callback(response, [], [], turn, '', {})
                new_messages = [{'role': 'user', 'content': TOOL_INTENT_WITHOUT_CALL_RETRY_PROMPT}]
                continue

            visible_reply = visible_reply or summary
            if on_event and visible_reply:
                on_event(agent_message_chunk(visible_reply))
            elif on_chunk and visible_reply and not plain_stream_emitted:
                on_chunk(visible_reply)
            elif assistant_replaced and on_chunk:
                on_chunk('\n\n' + response.content)
            elif assistant_replaced and not on_event and not on_chunk:
                print(response.content)
            exit_reason = {'result': 'NO_TOOL_CALL', 'data': response.content}
            handler.turn_end_callback(response, [], [], turn, '', exit_reason)
            exit_reason['usage'] = dict(total_usage)
            if on_event:
                on_event(done(stop_reason(exit_reason), usage=total_usage))
            return exit_reason

        if on_event:
            on_event(thought_done(
                model_step_id,
                content=_stream_preview_text(_model_process_content(response.content)),
            ))

        # 3) 顺序执行所有工具调用
        tool_results = []        # [{role:'tool', tool_call_id, content}]
        tool_result_meta = []     # [{tool_name, tool_call_id}]
        next_prompts = set()
        for ii, tc in enumerate(tool_calls):
            name, args, tid = tc['tool_name'], tc['args'], tc['id']
            tool_call_id = f'call_{turn}_{ii}'
            emit_tool_progress = name != 'ask_user'
            if on_event and emit_tool_progress:
                on_event(tool_call(tool_call_id, name, args))
                on_event(tool_call_update(tool_call_id, 'in_progress'))
            elif not on_event and not on_chunk:
                print(f'\n[Tool] {name}')
            try:
                if on_event and emit_tool_progress:
                    handler._tool_event_emit = lambda status, content='', data=None: on_event(
                        tool_call_update(tool_call_id, status, content, data=data)
                    )
                dispatch_args = dict(args)
                dispatch_args['_tool_call_id'] = tool_call_id
                outcome = handler.dispatch(name, dispatch_args, response, index=ii)
            except Exception as e:
                if on_event and emit_tool_progress:
                    on_event(tool_call_update(tool_call_id, 'failed', str(e)))
                raise
            finally:
                if hasattr(handler, '_tool_event_emit'):
                    handler._tool_event_emit = None
            result_text = stringify(outcome.data) or '(no output)'
            result_text = _persist_tool_result_if_needed(
                handler,
                result_text,
                name,
                tool_call_id,
            )
            if on_event and emit_tool_progress:
                status = 'failed' if isinstance(outcome.data, dict) and outcome.data.get('status') == 'error' else 'completed'
                content = '' if name == 'code_run' and isinstance(outcome.data, dict) else result_text
                on_event(tool_call_update(tool_call_id, status, content, data=_tool_event_data(outcome.data, result_text)))

            # OpenAI tool 响应消息：tool_call_id 必须与 assistant.tool_calls[i].id 对齐
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tid,
                'content': result_text,
            })
            tool_result_meta.append({
                'tool_name': name,
                'tool_call_id': tool_call_id,
            })

            if outcome.should_exit:
                exit_reason = {'result': 'EXITED', 'data': outcome.data}
                break
            if outcome.next_prompt is None:
                exit_reason = {'result': 'CURRENT_TASK_DONE', 'data': outcome.data}
                break
            next_prompts.add(outcome.next_prompt)

        _persist_assistant_response_if_needed(handler, client, response, model_step_id)

        # 3.5) 拦截任务完成出口：若 handler._done_hooks 还有待办，弹出注入下一轮
        # 参考 GenericAgent agent_loop.py:91-93。仅在 CURRENT_TASK_DONE / 无 next_prompt 时拦截，
        # EXITED（用户取消等强制退出）跳过；hook 队列空也跳过。
        if (not next_prompts) or exit_reason:
            hooks = getattr(handler, '_done_hooks', None) or []
            if hooks and exit_reason.get('result') != 'EXITED':
                next_prompts.add(hooks.pop(0))
                exit_reason = {}
                if on_event:
                    on_event(thought('Starting internal memory review.', title='Memory review'))

        _enforce_tool_result_budget(handler, tool_results, tool_result_meta)

        # 4) 触发 turn_end_callback 拼下一轮 user prompt
        joined = '\n'.join(next_prompts) if next_prompts else ''
        next_prompt = handler.turn_end_callback(response, tool_calls, tool_results,
                                                turn, joined, exit_reason)
        if exit_reason:
            exit_reason['usage'] = dict(total_usage)
            if on_event:
                on_event(done(stop_reason(exit_reason), usage=total_usage))
            return exit_reason

        # 5) 拼下一轮 new_messages：先 tool_results，再可选 user 文本提示
        new_messages = list(tool_results)
        if next_prompt and next_prompt.strip():
            new_messages.append({'role': 'user', 'content': next_prompt})

    fallback_response, _ = _run_max_turns_fallback(
        client,
        system_prompt,
        new_messages,
        max_turns,
        handler=handler,
        on_chunk=on_chunk,
        on_event=on_event,
    )
    _accumulate(fallback_response)
    exit_reason = {'result': 'MAX_TURNS_EXCEEDED', 'data': fallback_response.content}
    try:
        handler.turn_end_callback(fallback_response, [], [], max_turns + 1, '', exit_reason)
    except Exception:
        pass
    exit_reason['usage'] = dict(total_usage)
    if on_event:
        on_event(done(stop_reason(exit_reason), usage=total_usage))
    return exit_reason
