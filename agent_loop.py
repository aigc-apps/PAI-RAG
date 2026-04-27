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


def _model_process_content(content):
    thoughts, cleaned = split_model_content(content or '')
    parts = thoughts[:]
    if cleaned:
        parts.append(cleaned)
    return '\n\n'.join(parts).strip()


def _model_final_answer(content):
    _, cleaned = split_model_content(content or '')
    return cleaned


def agent_runner_loop(client, system_prompt, user_input, handler, tools_schema,
                      max_turns=40, on_chunk=None, on_event=None):
    """主循环。第一轮用 user_input 文本启动；之后用 tool_results + next_prompt。"""
    handler.max_turns = max_turns
    new_messages = [{'role': 'user', 'content': user_input}]
    exit_reason = {}

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
            if on_event:
                on_event(thought_done(
                    model_step_id,
                    hidden=True,
                    content=_model_process_content(response.content),
                ))
                cleaned = _model_final_answer(response.content)
                if cleaned:
                    on_event(agent_message_chunk(cleaned))
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
                outcome = handler.dispatch(name, args, response, index=ii)
            except Exception as e:
                if on_event and emit_tool_progress:
                    on_event(tool_call_update(tool_call_id, 'failed', str(e)))
                raise
            finally:
                if hasattr(handler, '_tool_event_emit'):
                    handler._tool_event_emit = None
            result_text = stringify(outcome.data) or '(no output)'
            if on_event and emit_tool_progress:
                status = 'failed' if isinstance(outcome.data, dict) and outcome.data.get('status') == 'error' else 'completed'
                content = '' if name == 'code_run' and isinstance(outcome.data, dict) else result_text
                on_event(tool_call_update(tool_call_id, status, content, data=outcome.data))

            # OpenAI tool 响应消息：tool_call_id 必须与 assistant.tool_calls[i].id 对齐
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tid,
                'content': result_text,
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
