"""精简版执行循环。
- 每轮：build new_messages → client.chat → 解析 tool_calls
- 工具结果按 OpenAI 原生 {'role':'tool', 'tool_call_id':...} 消息回填
- Web/ACP 通过结构化 AgentEvent 输出
- 退出条件：should_exit / 无 next_prompt / 达到 max_turns
"""
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
    '请基于已经完成的工具结果和证据，立即输出用户可见的最终报告正文。\n'
    '不要再调用工具，不要只输出 <summary>。如果需要保留摘要，只能放在报告正文之后。\n'
    '报告正文必须包含：诊断结论、日志证据、配置证据、因果链路、证据边界。'
)

TOOL_RESULT_MAX_CHARS = 100000
TURN_TOOL_RESULT_BUDGET_CHARS = 200000
TOOL_RESULT_PREVIEW_CHARS = 12000
PERSISTED_OUTPUT_TAG = '<persisted-output>'


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


def _persisted_output_message(content, path, reason):
    preview = _head_tail_text(content)
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
    if tool_name == 'file_read' or PERSISTED_OUTPUT_TAG in content:
        return content
    if not force and len(content) <= TOOL_RESULT_MAX_CHARS:
        return content

    path = None
    try:
        path = handler.persist_tool_result(content, tool_call_id, subdir='tool_results')
    except Exception as e:
        path = None
        reason = f'{reason or "tool result too large"}; persist failed: {type(e).__name__}: {e}'
    if not path:
        if force:
            return _head_tail_text(content)
        return content
    return _persisted_output_message(content, path, reason or 'tool result too large')


def _enforce_tool_result_budget(handler, tool_results, tool_result_meta):
    total = sum(len(msg.get('content') or '') for msg in tool_results)
    while total > TURN_TOOL_RESULT_BUDGET_CHARS:
        candidates = [
            (len(msg.get('content') or ''), index)
            for index, msg in enumerate(tool_results)
            if tool_result_meta[index].get('tool_name') != 'file_read'
            and PERSISTED_OUTPUT_TAG not in (msg.get('content') or '')
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


def agent_runner_loop(client, system_prompt, user_input, handler, tools_schema,
                      max_turns=40, on_chunk=None, on_event=None):
    """主循环。第一轮用 user_input 文本启动；之后用 tool_results + next_prompt。"""
    handler.max_turns = max_turns
    new_messages = [{'role': 'user', 'content': user_input}]
    exit_reason = {}
    summary_only_retry_used = False

    for turn in range(1, max_turns + 1):
        handler.current_turn = turn
        # 1) 调用 LLM
        gen = client.chat(system=system_prompt, new_messages=new_messages, tools=tools_schema)
        model_step_id = f'model-{turn}'
        model_raw_content = ''
        model_process_content = ''
        if on_event:
            on_event(thought_start(model_step_id, title='Agent step'))
        try:
            while True:
                chunk = next(gen)
                if isinstance(chunk, ToolCallDelta):
                    if on_event:
                        on_event(tool_call_delta(
                            f'tool-{turn}-{chunk.index}',
                            index=chunk.index,
                            name=chunk.name,
                            name_delta=chunk.name_delta,
                            arguments_delta=chunk.arguments_delta,
                            arguments_text=chunk.arguments_text,
                        ))
                    continue
                if on_event:
                    model_raw_content += chunk
                    next_process_content = stream_model_process_content(model_raw_content)
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
                    on_chunk(chunk)
                else:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
        except StopIteration as e:
            response = e.value
        if not on_event and not on_chunk:
            print()

        # 2) 解析 tool_calls
        tool_calls = [{'tool_name': tc.name, 'args': tc.input, 'id': tc.id}
                      for tc in response.tool_calls]
        if not tool_calls:
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

            if not visible_reply and summary and not summary_only_retry_used and turn < max_turns:
                summary_only_retry_used = True
                handler.turn_end_callback(response, [], [], turn, '', {})
                new_messages = [{'role': 'user', 'content': SUMMARY_ONLY_FINAL_RETRY_PROMPT}]
                continue

            visible_reply = visible_reply or summary
            if on_event and visible_reply:
                on_event(agent_message_chunk(visible_reply))
            exit_reason = {'result': 'NO_TOOL_CALL', 'data': response.content}
            handler.turn_end_callback(response, [], [], turn, '', exit_reason)
            if on_event:
                on_event(done(stop_reason(exit_reason)))
            return exit_reason

        if on_event:
            on_event(thought_done(
                model_step_id,
                content=_model_process_content(response.content),
            ))

        # 3) 顺序执行所有工具调用
        tool_results = []        # [{role:'tool', tool_call_id, content}]
        tool_result_meta = []     # [{tool_name, tool_call_id}]
        next_prompts = set()
        for ii, tc in enumerate(tool_calls):
            name, args, tid = tc['tool_name'], tc['args'], tc['id']
            tool_call_id = f'tool-{turn}-{ii}'
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
            if on_event:
                on_event(done(stop_reason(exit_reason)))
            return exit_reason

        # 5) 拼下一轮 new_messages：先 tool_results，再可选 user 文本提示
        new_messages = list(tool_results)
        if next_prompt and next_prompt.strip():
            new_messages.append({'role': 'user', 'content': next_prompt})

    exit_reason = {'result': 'MAX_TURNS_EXCEEDED'}
    if on_event:
        on_event(done(stop_reason(exit_reason)))
    return exit_reason
