"""MiniAgent API 客户端调用示例（纯标准库，零依赖）

覆盖 3 类接口：
  1. /v1/chat/completions   —— OpenAI Chat Completions 兼容（纯文本）
  2. /v1/responses          —— OpenAI Responses 兼容（结构化 + HITL，重点演示）
  3. /v1/sessions           —— 会话管理（含 cancel / pending_hitl）

运行：
    python scripts/client_examples.py --base-url http://127.0.0.1:8000
可单独跑某节：
    python scripts/client_examples.py --only responses-stream
    python scripts/client_examples.py --only responses-ask-user
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from contextlib import closing
from typing import Any, Callable, Iterator


# ────────────────────────────── HTTP / SSE 工具 ────────────────────────────── #

class APIError(RuntimeError):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        message = body.get('error', {}).get('message') if isinstance(body, dict) else str(body)
        super().__init__(f'{status}: {message}')


def http_request(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    stream: bool = False,
    timeout: int = 300,
):
    """统一 HTTP 入口。stream=True 时返回 (status, headers, urlopen_handle)；否则 (status, headers, json_body)。"""
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read().decode('utf-8', errors='replace')
        try:
            body = json.loads(raw)
        except Exception:
            body = raw
        raise APIError(e.code, body) from None

    norm_headers = {k.lower(): v for k, v in resp.headers.items()}
    if stream:
        return resp.status, norm_headers, resp
    raw = resp.read().decode('utf-8', errors='replace')
    body_obj = json.loads(raw) if raw else None
    resp.close()
    return resp.status, norm_headers, body_obj


def iter_sse(resp) -> Iterator[tuple[str, str]]:
    """从 SSE 响应里迭代出 (event_name, data_text) 元组。"""
    event_name = ''
    data_buf: list[str] = []

    def flush():
        nonlocal event_name, data_buf
        if data_buf:
            payload = '\n'.join(data_buf)
            yield event_name, payload
        event_name = ''
        data_buf = []

    with closing(resp):
        for raw in resp:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            if line == '':
                yield from flush()
                continue
            if line.startswith(':'):
                continue
            if line.startswith('event:'):
                event_name = line[len('event:'):].strip()
                continue
            if line.startswith('data:'):
                data_buf.append(line[len('data:'):].lstrip())
        yield from flush()


# ─────────────────────────── 1. Chat Completions ─────────────────────────── #

def demo_chat_nonstream(base_url: str, model: str) -> None:
    print('--- /v1/chat/completions (stream=false) ---')
    status, _, body = http_request(
        'POST', f'{base_url}/v1/chat/completions',
        body={
            'model': model,
            'messages': [{'role': 'user', 'content': '用一句话介绍你自己'}],
            'stream': False,
        },
    )
    print(f'status: {status}')
    print(f'message: {body["choices"][0]["message"]["content"]}')
    print(f'usage:   {body["usage"]}')


def demo_chat_stream(base_url: str, model: str) -> None:
    print('--- /v1/chat/completions (stream=true) ---')
    _, headers, resp = http_request(
        'POST', f'{base_url}/v1/chat/completions',
        body={
            'model': model,
            'messages': [{'role': 'user', 'content': '用一句话介绍你自己'}],
            'stream': True,
        },
        stream=True,
    )
    print(f'session: {headers.get("x-session-id")}')
    final_usage = None
    sys.stdout.write('message: ')
    sys.stdout.flush()
    for event_name, data in iter_sse(resp):
        if data == '[DONE]':
            break
        chunk = json.loads(data)
        for choice in chunk.get('choices', []):
            delta = choice.get('delta') or {}
            if delta.get('content'):
                sys.stdout.write(delta['content'])
                sys.stdout.flush()
        if chunk.get('usage'):
            final_usage = chunk['usage']
    sys.stdout.write('\n')
    print(f'usage:   {final_usage}')


# ──────────────────────────── 2. Responses API ──────────────────────────── #

def render_response_event(event_name: str, payload: dict, state: dict) -> None:
    """把 /v1/responses SSE 事件按类型漂亮打印。"""
    et = payload.get('type', event_name)

    if et == 'response.created':
        state['response_id'] = payload.get('id', '')
        print(f'[resp] created id={state["response_id"]} model={payload.get("model")}')

    elif et == 'response.output_item.added':
        item = payload.get('item') or {}
        if item.get('type') == 'function_call':
            name = item.get('name', '?')
            call_id = item.get('call_id', '?')
            print(f'[resp] tool.start call_id={call_id} name={name}')
            state.setdefault('tools', {})[call_id] = {'name': name, 'args': ''}
        elif item.get('type') == 'function_call_output':
            call_id = item.get('call_id', '?')
            out = (item.get('output') or '')[:120]
            print(f'[resp] tool.output call_id={call_id} output={out!r}')

    elif et == 'response.function_call_arguments.delta':
        # 工具参数增量；演示里不打印每片，只累计
        item_id = payload.get('item_id', '')
        state.setdefault('args_buf', {}).setdefault(item_id, '')
        state['args_buf'][item_id] += payload.get('delta', '')

    elif et == 'response.function_call_arguments.done':
        item_id = payload.get('item_id', '')
        args = payload.get('arguments') or state.get('args_buf', {}).get(item_id, '')
        print(f'[resp] tool.args  item_id={item_id} args={args!r}')

    elif et == 'response.output_text.delta':
        sys.stdout.write(payload.get('delta', ''))
        sys.stdout.flush()
        state.setdefault('final_text_parts', []).append(payload.get('delta', ''))

    elif et == 'response.reasoning_step.started':
        # 合成的思考步骤边界（前缀 rs_synth_）
        pass

    elif et == 'response.reasoning_step.completed':
        pass

    elif et == 'response.requires_action':
        # HITL 中断点：把 ask_user 信息存下来交给上层
        ra = payload.get('required_action') or {}
        sub = ra.get('submit_tool_outputs') or {}
        calls = sub.get('tool_calls') or []
        if calls:
            tc = calls[0]
            fn = tc.get('function') or {}
            args = {}
            try:
                args = json.loads(fn.get('arguments') or '{}')
            except Exception:
                pass
            state['pending_hitl'] = {
                'response_id': payload.get('id', ''),
                'call_id': tc.get('id', ''),
                'tool_name': fn.get('name', ''),
                'question': args.get('question'),
                'candidates': args.get('candidates'),
            }
            print(f'\n[resp] requires_action  response_id={payload.get("id")} '
                  f'tool={fn.get("name")} question={args.get("question")!r}')

    elif et == 'response.completed':
        if state.get('final_text_parts'):
            sys.stdout.write('\n')
        usage = payload.get('usage')
        print(f'[resp] completed  status={payload.get("status")} usage={usage}')
        state['terminal'] = True

    elif et == 'response.failed':
        err = payload.get('error') or {}
        print(f'[resp] failed     {err}')
        state['terminal'] = True


def demo_responses_nonstream(base_url: str, model: str) -> None:
    print('--- /v1/responses (stream=false) ---')
    status, _, body = http_request(
        'POST', f'{base_url}/v1/responses',
        body={
            'model': model,
            'input': '用 ls 列一下当前目录前 2 个文件',
            'stream': False,
        },
    )
    print(f'status:   {status}')
    print(f'resp_id:  {body["id"]}')
    print(f'output items:')
    for item in body.get('output') or []:
        if item['type'] == 'function_call':
            print(f'  - function_call  name={item["name"]} call_id={item["call_id"]} args={item["arguments"]}')
        elif item['type'] == 'function_call_output':
            print(f'  - function_call_output  call_id={item["call_id"]} output={item["output"][:80]}...')
        elif item['type'] == 'message':
            text = ''.join(c.get('text', '') for c in item['content'])
            print(f'  - message  text={text}')
    print(f'usage:    {body.get("usage")}')


def demo_responses_stream_multi_turn(base_url: str, model: str) -> None:
    """单一 session_id 多轮：每轮独立 POST /v1/responses，复用同一个 session_id。"""
    print('--- /v1/responses (stream=true) 多轮，复用 session_id ---')

    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']
    print(f'session_id: {sid}')

    def stream_one(req_body: dict, label: str) -> dict:
        print(f'\n[{label}] request: {req_body["input"]!r}')
        _, _, resp = http_request(
            'POST', f'{base_url}/v1/responses',
            body=req_body, stream=True,
        )
        state: dict = {}
        for event_name, data in iter_sse(resp):
            if data == '[DONE]':
                break
            payload = json.loads(data)
            render_response_event(event_name, payload, state)
            if state.get('terminal'):
                break
        return state

    def with_retry(req_body: dict, label: str) -> dict:
        # 上一轮 SSE 刚结束时，session 可能还没 idle；遇到 409 退避
        for attempt in range(6):
            try:
                return stream_one(req_body, label)
            except APIError as e:
                if e.status == 409:
                    time.sleep(0.5)
                    continue
                raise
        return {}

    with_retry(
        {'session_id': sid, 'model': model, 'input': '记住一个数字：7。然后回复"好的"', 'stream': True},
        'turn-1',
    )
    with_retry(
        {'session_id': sid, 'model': model, 'input': '我刚才让你记的数字是什么？', 'stream': True},
        'turn-2',
    )


def demo_responses_ask_user(base_url: str, model: str) -> None:
    """演示 HITL：requires_action → 客户端用 previous_response_id + function_call_output 续传。"""
    print('--- /v1/responses ask_user 中断/恢复 ---')
    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']

    def consume(req_body: dict, label: str) -> dict:
        print(f'\n[{label}] body: {req_body}')
        _, _, resp = http_request(
            'POST', f'{base_url}/v1/responses', body=req_body, stream=True,
        )
        state: dict = {}
        for event_name, data in iter_sse(resp):
            if data == '[DONE]':
                break
            payload = json.loads(data)
            render_response_event(event_name, payload, state)
            if state.get('terminal'):
                break
        return state

    prompt = (
        '请帮我新建一个文本文件，文件名我没告诉你。'
        '在确定文件名之前先用 ask_user 工具问我"想叫什么名字"，'
        '拿到回答后再继续写入。'
    )
    state = consume(
        {'session_id': sid, 'model': model, 'input': prompt, 'stream': True},
        'turn-1',
    )

    pending = state.get('pending_hitl')
    if not pending:
        print('[client] Agent 这次没问问题，正常结束')
        return

    print(f'\n[client] Agent 问："{pending["question"]}" → 续传答复')
    consume(
        {
            'previous_response_id': pending['response_id'],
            'input': [{
                'type': 'function_call_output',
                'call_id': pending['call_id'],
                'output': '就叫 hello.txt 吧，内容写 hello world',
            }],
            'stream': True,
        },
        'turn-1-answer',
    )


def demo_responses_cancel(base_url: str, model: str) -> None:
    """演示 cancel：发起一个长 response，立即调用 /v1/sessions/{id}/cancel。"""
    print('--- /v1/responses cancel ---')
    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']

    # 1) 起一个流；用 thread 跑，主线程 sleep 后 cancel
    import threading

    def runner():
        try:
            _, _, resp = http_request(
                'POST', f'{base_url}/v1/responses',
                body={
                    'session_id': sid, 'model': model,
                    'input': '请详细分析一下 backend/server.py 的所有路由，每条都给出文件读出的代码片段',
                    'stream': True,
                },
                stream=True,
            )
            for event_name, data in iter_sse(resp):
                if data == '[DONE]':
                    break
                payload = json.loads(data)
                render_response_event(event_name, payload, {})
        except APIError as e:
            print(f'[runner] APIError {e}')

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(1.5)
    print('\n[client] firing cancel')
    _, _, _ = http_request('POST', f'{base_url}/v1/sessions/{sid}/cancel')
    t.join(timeout=10)
    _, _, sess_now = http_request('GET', f'{base_url}/v1/sessions/{sid}')
    print(f'[client] session status after cancel: {sess_now.get("status")}')


# ──────────────────────────── 3. Sessions API ──────────────────────────── #

def demo_sessions(base_url: str) -> None:
    print('--- /v1/sessions ---')
    _, _, created = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = created['session_id']
    print(f'created: {created}')

    _, _, listed = http_request('GET', f'{base_url}/v1/sessions')
    print(f'list count: {len(listed["data"])}')

    _, _, got = http_request('GET', f'{base_url}/v1/sessions/{sid}')
    print(f'get: status={got["status"]} pending_hitl={got.get("pending_hitl")!r}')

    _, _, deleted = http_request('DELETE', f'{base_url}/v1/sessions/{sid}')
    print(f'deleted: {deleted}')


# ──────────────────────────────── main ──────────────────────────────── #

DEMOS: dict[str, Callable[..., None]] = {
    'models':              lambda url, model: _show_models(url),
    'chat':                lambda url, model: demo_chat_nonstream(url, model),
    'chat-stream':         lambda url, model: demo_chat_stream(url, model),
    'responses':           lambda url, model: demo_responses_nonstream(url, model),
    'responses-stream':    lambda url, model: demo_responses_stream_multi_turn(url, model),
    'responses-ask-user':  lambda url, model: demo_responses_ask_user(url, model),
    'responses-cancel':    lambda url, model: demo_responses_cancel(url, model),
    'sessions':            lambda url, model: demo_sessions(url),
}


def _show_models(base_url: str) -> None:
    print('--- /v1/models ---')
    _, _, body = http_request('GET', f'{base_url}/v1/models')
    for m in body['data']:
        print(f'  {m["id"]:20s} created={m["created"]} owned_by={m["owned_by"]}')


def main():
    parser = argparse.ArgumentParser(description='MiniAgent API 调用示例')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen-plus')
    parser.add_argument('--only', choices=list(DEMOS), help='只跑指定的一节')
    args = parser.parse_args()

    if args.only:
        DEMOS[args.only](args.base_url, args.model)
        return

    for name in [
        'models', 'chat', 'chat-stream', 'responses',
        'responses-stream', 'responses-ask-user', 'sessions',
    ]:
        print()
        print('=' * 70)
        print(f'  {name}')
        print('=' * 70)
        try:
            DEMOS[name](args.base_url, args.model)
        except APIError as e:
            print(f'!! {name} failed: {e}')


if __name__ == '__main__':
    main()
