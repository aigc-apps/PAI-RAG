"""精简版执行循环。
- 每轮：build new_messages → client.chat 流式输出 → 解析 tool_calls
- 工具结果按 OpenAI 原生 {'role':'tool', 'tool_call_id':...} 消息回填
- 退出条件：should_exit / 无 next_prompt / 达到 max_turns
"""
import json, sys
from dataclasses import dataclass
from typing import Any, Optional


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


def _stringify(data):
    if data is None:
        return ''
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, default=str)
    return str(data)


def agent_runner_loop(client, system_prompt, user_input, handler, tools_schema,
                      max_turns=40, on_chunk=None):
    """主循环。第一轮用 user_input 文本启动；之后用 tool_results + next_prompt。"""
    handler.max_turns = max_turns
    new_messages = [{'role': 'user', 'content': user_input}]
    exit_reason = {}

    for turn in range(1, max_turns + 1):
        handler.current_turn = turn
        if on_chunk:
            on_chunk(f'\n── Turn {turn} ──\n')
        else:
            print(f'\n\033[1m── Turn {turn} ──\033[0m')

        # 1) 调用 LLM（流式打印文本）
        gen = client.chat(system=system_prompt, new_messages=new_messages, tools=tools_schema)
        try:
            while True:
                chunk = next(gen)
                if on_chunk:
                    on_chunk(chunk)
                else:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
        except StopIteration as e:
            response = e.value
        if not on_chunk:
            print()

        # 2) 解析 tool_calls
        tool_calls = [{'tool_name': tc.name, 'args': tc.input, 'id': tc.id}
                      for tc in response.tool_calls]
        if not tool_calls:
            if on_chunk:
                on_chunk('\n[Info] 模型未调用工具，任务结束。\n')
            else:
                print('[Info] 模型未调用工具，任务结束。')
            exit_reason = {'result': 'NO_TOOL_CALL', 'data': response.content}
            handler.turn_end_callback(response, [], [], turn, '', exit_reason)
            return exit_reason

        # 3) 顺序执行所有工具调用
        tool_results = []        # [{role:'tool', tool_call_id, content}]
        next_prompts = set()
        for ii, tc in enumerate(tool_calls):
            name, args, tid = tc['tool_name'], tc['args'], tc['id']
            if name == 'ask_user':
                tool_info = f"\n🛠️  {name}"
            else:
                tool_info = f"\n🛠️  {name}({json.dumps({k:v for k,v in args.items() if k!='_index'}, ensure_ascii=False)[:120]})"
            if on_chunk:
                on_chunk(tool_info + '\n')
            else:
                print(tool_info)
            outcome = handler.dispatch(name, args, response, index=ii)

            # OpenAI tool 响应消息：tool_call_id 必须与 assistant.tool_calls[i].id 对齐
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tid,
                'content': _stringify(outcome.data) or '(no output)',
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
                # UI 哨兵：标记下一轮为内部 turn（前端可识别后折叠展示）
                if on_chunk:
                    on_chunk('\n[[INTERNAL_TURN_START]]\n')

        # 4) 触发 turn_end_callback 拼下一轮 user prompt
        joined = '\n'.join(next_prompts) if next_prompts else ''
        next_prompt = handler.turn_end_callback(response, tool_calls, tool_results,
                                                turn, joined, exit_reason)
        if exit_reason:
            return exit_reason

        # 5) 拼下一轮 new_messages：先 tool_results，再可选 user 文本提示
        new_messages = list(tool_results)
        if next_prompt and next_prompt.strip():
            new_messages.append({'role': 'user', 'content': next_prompt})

    return {'result': 'MAX_TURNS_EXCEEDED'}
