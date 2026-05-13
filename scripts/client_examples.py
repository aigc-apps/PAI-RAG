"""MiniAgent API 客户端调用示例（纯标准库，零依赖）

覆盖 4 类接口：
  1. /v1/chat/completions   —— OpenAI Chat Completions 兼容（纯文本）
  2. /v1/responses          —— OpenAI Responses 兼容（结构化输出）
  3. /v1/runs               —— 自定义 Agent 生命周期（重点演示）
  4. /v1/sessions           —— 会话管理

运行：
    python scripts/client_examples.py --base-url http://127.0.0.1:8765
可单独跑某节：
    python scripts/client_examples.py --only runs
    python scripts/client_examples.py --only chat --stream
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

    # urllib 给的 HTTPMessage 已经 case-insensitive，但 dict() 后丢失这个能力。
    # 我们包成 lowercase dict，调用方一律用小写键访问。
    norm_headers = {k.lower(): v for k, v in resp.headers.items()}
    if stream:
        return resp.status, norm_headers, resp
    raw = resp.read().decode('utf-8', errors='replace')
    body_obj = json.loads(raw) if raw else None
    resp.close()
    return resp.status, norm_headers, body_obj


def iter_sse(resp) -> Iterator[tuple[str, str]]:
    """从 SSE 响应里迭代出 (event_name, data_text) 元组。

    服务端可能发送：
      - 注释行 `: keepalive`（忽略）
      - `event: X` + `data: Y` 配对
      - 仅 `data: Y`（默认事件，event 为空串）
      - `data: [DONE]` 表示流结束
    """
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
                continue  # comment / keepalive
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
    print(f'session: {body["metadata"]["session_id"]}')


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
        # 每条 chat.completion.chunk
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
    print('--- /v1/responses (stream=true) + previous_response_id 多轮 ---')

    def stream_one(req_body: dict, label: str) -> str:
        print(f'\n[{label}] request: {req_body["input"]!r}')
        _, _, resp = http_request(
            'POST', f'{base_url}/v1/responses',
            body=req_body, stream=True,
        )
        text_parts, resp_id = [], ''
        for event_name, data in iter_sse(resp):
            if data == '[DONE]':
                break
            payload = json.loads(data)
            t = payload.get('type', event_name)
            if t == 'response.created':
                resp_id = payload['id']
                print(f'[{label}] resp_id: {resp_id}')
            elif t == 'response.output_text.delta':
                text_parts.append(payload.get('delta', ''))
            elif t == 'response.completed':
                print(f'[{label}] status: {payload["status"]}, usage: {payload.get("usage")}')
        print(f'[{label}] text: {"".join(text_parts)}')
        return resp_id

    rid = stream_one({'model': model, 'input': '记住一个数字：7。然后回复"好的"', 'stream': True}, 'turn-1')
    # SSE 流刚结束时，服务端 session 状态可能还没标记 idle，下一轮可能会 409 session_busy。
    # 简单做法是带重试：碰到 session_busy 时退避后再试。
    body = {'model': model, 'previous_response_id': rid, 'input': '我刚才让你记的数字是什么？', 'stream': True}
    for attempt in range(6):
        try:
            stream_one(body, 'turn-2')
            break
        except APIError as e:
            if e.status == 409:
                time.sleep(0.5)
                continue
            raise


# ───────────────────────────── 3. Runs API（重点）───────────────────────── #

def render_run_event(event: dict, state: dict) -> None:
    """按事件类型把 SSE event 漂亮打印到 stdout。

    state 是调用方持有的累计上下文（拼最终文本、记录工具状态等）。
    """
    et = event.get('event')
    rid = event.get('run_id', '')
    short = rid[-6:] if rid else ''
    ts = event.get('timestamp')
    prefix = f'[{short}]'

    if et == 'reasoning.started':
        print(f'{prefix} reasoning.started   step={event["step_id"]} title={event.get("title")!r}')

    elif et == 'reasoning.available':
        text = event.get('text', '')
        replace = event.get('replace')
        marker = 'REPLACE' if replace else 'append'
        print(f'{prefix} reasoning.available step={event["step_id"]} [{marker}] {text!r}')

    elif et == 'reasoning.completed':
        print(f'{prefix} reasoning.completed step={event["step_id"]} status={event.get("status")}')

    elif et == 'tool.delta':
        # 工具参数流式增量。一般 UI 只展示 arguments_text 累计快照
        print(f'{prefix} tool.delta          call={event["tool_call_id"]} tool={event["tool"]} '
              f'args_so_far={event.get("arguments_text", "")!r}')

    elif et == 'tool.started':
        state.setdefault('tools', {})[event['tool_call_id']] = {
            'tool': event['tool'], 'input': event.get('input'), 'started_at': ts,
        }
        print(f'{prefix} tool.started        call={event["tool_call_id"]} tool={event["tool"]} '
              f'input={event.get("input")}')

    elif et == 'tool.updated':
        print(f'{prefix} tool.updated        call={event["tool_call_id"]} status={event.get("status")}')

    elif et == 'tool.completed':
        meta = state.get('tools', {}).get(event['tool_call_id'], {})
        elapsed = (ts - meta['started_at']) if meta.get('started_at') and ts else None
        content = (event.get('content') or '')[:120]
        print(f'{prefix} tool.completed      call={event["tool_call_id"]} status={event.get("status")} '
              f'elapsed={elapsed:.2f}s content={content!r}' if elapsed else
              f'{prefix} tool.completed      call={event["tool_call_id"]} status={event.get("status")} '
              f'content={content!r}')

    elif et == 'message.delta':
        state.setdefault('final_text_parts', []).append(event.get('delta', ''))
        # 流式渲染：把每个 delta 直接打到 stdout（实际 UI 里就是逐字打字效果）
        sys.stdout.write(event.get('delta', ''))
        sys.stdout.flush()

    elif et == 'ask_user':
        # Agent 中断等待用户输入
        state['ask_user'] = event
        print(f'\n{prefix} ask_user            question={event.get("question")!r} '
              f'candidates={event.get("candidates")}')

    elif et == 'run.completed':
        if state.get('final_text_parts'):
            sys.stdout.write('\n')
        print(f'{prefix} run.completed       output_len={len(event.get("output", ""))} '
              f'usage={event.get("usage")}')
        state['completed'] = True

    elif et == 'run.failed':
        print(f'{prefix} run.failed          error={event.get("error")!r}')
        state['completed'] = True

    else:
        print(f'{prefix} {et}  {event}')


def demo_runs_two_stage(base_url: str) -> None:
    """两段式：POST 创建 → GET 订阅事件。适合需要异步执行、断点续传的场景。"""
    print('--- /v1/runs (两段式：POST 创建 → GET events) ---')

    # 第 1 步：先创建 session（可选；不传 session_id 时服务端会自动新建一个）
    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']
    print(f'session_id: {sid}')

    # 第 2 步：创建 run
    _, _, run = http_request(
        'POST', f'{base_url}/v1/runs',
        body={'session_id': sid, 'input': '用 ls 列一下当前目录前 2 个文件，然后用一句话总结'},
    )
    run_id = run['run_id']
    cursor = run['cursor']
    print(f'run_id: {run_id}, cursor: {cursor}, status: {run["status"]}')

    # 第 3 步：订阅事件（可用 ?last_event_id=<cursor> 从断点续传；首次传 "0-0" 或省略表示从头）
    _, _, resp = http_request(
        'GET', f'{base_url}/v1/runs/{run_id}/events',
        stream=True,
    )
    state: dict = {}
    last_event_id = cursor
    for event_name, data in iter_sse(resp):
        if data == '[DONE]':
            break
        event = json.loads(data)
        render_run_event(event, state)
        # 实际重连场景：每条事件其实有自己的 SSE id（服务端通过 id: 行下发），
        # 这里演示用 timestamp 模拟；生产里直接读 SSE 的 Last-Event-ID。
        if state.get('completed'):
            break

    # 第 4 步：查询 run 终态。SSE 流结束和服务端把 run 标记为 completed/failed 之间有一个小窗口，
    # 所以做几次轮询直到拿到终态。
    for _ in range(20):
        _, _, status_obj = http_request('GET', f'{base_url}/v1/runs/{run_id}')
        if status_obj['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(0.3)
    print(f'final run state: status={status_obj["status"]} finished_at={status_obj["finished_at"]}')


def demo_runs_one_shot_stream(base_url: str) -> None:
    """一段式：POST stream=true 直接拿 SSE 流。最常用、最低延迟。"""
    print('\n--- /v1/runs (一段式：POST stream=true) ---')
    _, headers, resp = http_request(
        'POST', f'{base_url}/v1/runs',
        body={'input': '用一句话介绍 README.md 是干什么的（用 file_read 读一下）', 'stream': True},
        stream=True,
    )
    print(f'session: {headers.get("x-session-id")}, run: {headers.get("x-run-id")}')
    state: dict = {}
    for event_name, data in iter_sse(resp):
        if data == '[DONE]':
            break
        event = json.loads(data)
        render_run_event(event, state)
        if state.get('completed'):
            break


def demo_runs_multi_turn(base_url: str) -> None:
    """多轮：复用同一个 session_id，每轮一个 run。Agent 看得到上轮历史。"""
    print('\n--- /v1/runs 多轮（同 session_id） ---')
    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']

    def run_once(text: str, label: str) -> None:
        print(f'\n[{label}] input={text!r}')
        _, _, resp = http_request(
            'POST', f'{base_url}/v1/runs',
            body={'session_id': sid, 'input': text, 'stream': True},
            stream=True,
        )
        for _, data in iter_sse(resp):
            if data == '[DONE]':
                break
            event = json.loads(data)
            if event.get('event') == 'message.delta':
                sys.stdout.write(event.get('delta', ''))
                sys.stdout.flush()
            elif event.get('event') == 'run.completed':
                print(f'\n[{label}] usage={event.get("usage")}')
                break
            elif event.get('event') == 'run.failed':
                print(f'\n[{label}] FAILED {event.get("error")}')
                break

    def run_with_retry(text: str, label: str) -> None:
        # session 上一轮 SSE 刚结束时可能还没 idle，409 后退避重试
        for attempt in range(6):
            try:
                run_once(text, label)
                return
            except APIError as e:
                if e.status == 409:
                    time.sleep(0.5)
                    continue
                raise

    run_with_retry('记住一个数字：42。然后只回复"好的"。', 'turn-1')
    run_with_retry('我刚才让你记的数字是什么？只说数字。', 'turn-2')


def demo_runs_ask_user(base_url: str) -> None:
    """演示 ask_user 中断 + 用户补充输入恢复。

    关键客户端模式：收到 `ask_user` 事件不要立刻断开。继续读到 `run.completed`，
    此时 server 才把 session 状态正式切到 `waiting_user`，下一轮 POST 才能稳定走
    "answer 当前 run" 而不是 "新建 run"。如果你提前断开，server 端 session 可能还在
    `running`，紧接着发回答会触发 409 session_busy。
    """
    print('\n--- /v1/runs ask_user 中断/恢复 ---')
    _, _, sess = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = sess['session_id']

    def consume(input_text: str, label: str) -> dict:
        _, _, resp = http_request(
            'POST', f'{base_url}/v1/runs',
            body={'session_id': sid, 'input': input_text, 'stream': True},
            stream=True,
        )
        state: dict = {}
        print(f'[{label}] input={input_text!r}')
        for _, data in iter_sse(resp):
            if data == '[DONE]':
                break
            event = json.loads(data)
            render_run_event(event, state)
            # 重要：即使收到 ask_user，也继续读到 run.completed 再退出循环
            if state.get('completed'):
                break
        return state

    # 这条 prompt 故意制造歧义 + 显式要求用 ask_user 工具问清楚，可靠触发中断。
    prompt = (
        '请帮我新建一个文本文件，文件名我没告诉你。'
        '在确定文件名之前先用 ask_user 工具问我"想叫什么名字"，'
        '拿到回答后再继续写入。'
    )
    state = consume(prompt, 'turn-1')
    if state.get('ask_user'):
        question = state['ask_user'].get('question', '')
        print(f'\n[client] Agent 问了："{question}"，向同一个 session 发回答')
        consume('就叫 hello.txt 吧，内容写 hello world', 'turn-1-answer')
    else:
        print('[client] Agent 这次没问问题，正常结束')


def demo_runs_stop(base_url: str) -> None:
    """演示 stop：发起一个长 run 立刻 stop 掉。"""
    print('\n--- /v1/runs stop ---')
    _, _, run = http_request(
        'POST', f'{base_url}/v1/runs',
        body={'input': '请详细分析一下 backend/server.py 的所有路由，每条都给出文件读出的代码片段'},
    )
    run_id = run['run_id']
    print(f'started run_id={run_id}, sleeping 1s then stop')
    time.sleep(1)
    _, _, stop_resp = http_request('POST', f'{base_url}/v1/runs/{run_id}/stop')
    print(f'stop response: {stop_resp}')
    # 终态等待几秒
    for _ in range(10):
        _, _, st = http_request('GET', f'{base_url}/v1/runs/{run_id}')
        if st['status'] in ('completed', 'failed', 'cancelled'):
            print(f'final status: {st["status"]}')
            return
        time.sleep(0.5)
    print('still running after 5s, gave up checking')


# ──────────────────────────── 4. Sessions API ──────────────────────────── #

def demo_sessions(base_url: str) -> None:
    print('--- /v1/sessions ---')
    _, _, created = http_request('POST', f'{base_url}/v1/sessions', body={})
    sid = created['session_id']
    print(f'created: {created}')

    _, _, listed = http_request('GET', f'{base_url}/v1/sessions')
    print(f'list count: {len(listed["data"])}')

    _, _, got = http_request('GET', f'{base_url}/v1/sessions/{sid}')
    print(f'get: status={got["status"]} active_run_id={got["active_run_id"]!r}')

    _, _, deleted = http_request('DELETE', f'{base_url}/v1/sessions/{sid}')
    print(f'deleted: {deleted}')


# ──────────────────────────────── main ──────────────────────────────── #

DEMOS: dict[str, Callable[..., None]] = {
    'models':      lambda url, model: _show_models(url),
    'chat':        lambda url, model: demo_chat_nonstream(url, model),
    'chat-stream': lambda url, model: demo_chat_stream(url, model),
    'responses':   lambda url, model: demo_responses_nonstream(url, model),
    'responses-stream': lambda url, model: demo_responses_stream_multi_turn(url, model),
    'runs':            lambda url, model: demo_runs_two_stage(url),
    'runs-stream':     lambda url, model: demo_runs_one_shot_stream(url),
    'runs-multi-turn': lambda url, model: demo_runs_multi_turn(url),
    'runs-ask-user':   lambda url, model: demo_runs_ask_user(url),
    'runs-stop':       lambda url, model: demo_runs_stop(url),
    'sessions':        lambda url, model: demo_sessions(url),
}


def _show_models(base_url: str) -> None:
    print('--- /v1/models ---')
    _, _, body = http_request('GET', f'{base_url}/v1/models')
    for m in body['data']:
        print(f'  {m["id"]:20s} created={m["created"]} owned_by={m["owned_by"]}')


def main():
    parser = argparse.ArgumentParser(description='MiniAgent API 调用示例')
    parser.add_argument('--base-url', default='http://127.0.0.1:8765')
    parser.add_argument('--model', default='qwen-plus')
    parser.add_argument('--only', choices=list(DEMOS), help='只跑指定的一节')
    args = parser.parse_args()

    if args.only:
        DEMOS[args.only](args.base_url, args.model)
        return

    # 默认按顺序跑一遍
    for name in ['models', 'chat', 'chat-stream', 'responses', 'runs-stream', 'runs', 'runs-ask-user', 'sessions']:
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
