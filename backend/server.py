import json
import os
import queue
import re
import threading
import time
import uuid

import anyio
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agent_service import (
    SESSION_WAITING_USER,
    AgentService,
    NoRegeneratableAnswerError,
    ServiceCapacityError,
    SessionBusyError,
    last_user_text,
    openai_chat_chunk,
    openai_chat_completion,
    openai_done_chunk,
    openai_role_chunk,
)
from backend.agent_service import handler_memory_scope
from backend.skills_inventory import skills_inventory
from backend.workspace import WorkspaceViolation
from session_store import SERVER_USER_ID
from tools import WorkspaceViolation as ToolWorkspaceViolation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import settings as config
import runtime_config


app = FastAPI(title='PAI-RAG OpenAI Compatible Backend')
service = AgentService()
RUNNER_BACKEND = (os.environ.get('RUNNER_BACKEND') or getattr(config, 'RUNNER_BACKEND', 'thread') or 'thread').lower()
celery_service = None
if RUNNER_BACKEND == 'celery':
    from backend.celery_runner import CeleryRunService

    celery_service = CeleryRunService(service.store, service.workspace_manager)
SSE_HEARTBEAT_SECONDS = int(getattr(config, 'SSE_HEARTBEAT_SECONDS', 15))
MAX_REQUEST_BODY_BYTES = int(getattr(config, 'MAX_REQUEST_BODY_BYTES', 8 * 1024 * 1024))
THREAD_RUNS = {}
THREAD_RUNS_LOCK = threading.RLock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, 'SERVER_CORS_ORIGINS', ['*']),
    allow_methods=['*'],
    allow_headers=['*'],
)


def _error_payload_from_detail(detail, default_code=''):
    if isinstance(detail, dict):
        payload = {
            'message': detail.get('message') or detail.get('detail') or '',
            'type': detail.get('type') or 'invalid_request_error',
            'code': detail.get('code') or default_code,
        }
        for key, value in detail.items():
            if key not in ('message', 'type', 'code', 'detail'):
                payload[key] = value
        return payload
    return {
        'message': '' if detail is None else str(detail),
        'type': 'invalid_request_error',
        'code': default_code,
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
    return JSONResponse({'error': _error_payload_from_detail(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    try:
        message = '; '.join(
            f"{'.'.join(str(part) for part in err.get('loc') or [])}: {err.get('msg') or ''}".strip(': ')
            for err in exc.errors()
        ) or 'Invalid request'
    except Exception:
        message = 'Invalid request'
    return JSONResponse(
        {'error': {'message': message, 'type': 'invalid_request_error', 'code': 'validation_error'}},
        status_code=422,
    )


@app.middleware('http')
async def http_guards(request: Request, call_next):
    content_length = request.headers.get('content-length')
    if MAX_REQUEST_BODY_BYTES and content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {'error': {'code': 'request_too_large', 'message': 'Request body is too large'}},
                    status_code=413,
                )
        except ValueError:
            pass
    response = await call_next(request)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    return response


def backend_error(exc):
    if isinstance(exc, SessionBusyError):
        return HTTPException(
            status_code=409,
            detail={'code': 'session_busy', 'message': str(exc), 'status': exc.status},
        )
    if isinstance(exc, ServiceCapacityError):
        return HTTPException(
            status_code=429,
            detail={'code': 'capacity_exceeded', 'message': str(exc), 'scope': exc.scope, 'limit': exc.limit},
        )
    if isinstance(exc, NoRegeneratableAnswerError):
        return HTTPException(
            status_code=409,
            detail={'code': 'no_regeneratable_answer', 'message': str(exc)},
        )
    if isinstance(exc, (WorkspaceViolation, ToolWorkspaceViolation)):
        return HTTPException(status_code=400, detail={'code': 'workspace_violation', 'message': str(exc)})
    return exc


def openai_error(message, code='invalid_request_error'):
    return {'error': {'message': message, 'type': 'invalid_request_error', 'code': code}}


def stream_headers(session_id, run_id=''):
    return {
        'X-Session-Id': session_id,
        'X-Run-Id': run_id or '',
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
    }


def sse_encode(data=None, event=None, event_id=None, comment=None):
    if comment is not None:
        return f': {comment}\n\n'
    lines = []
    if event_id:
        lines.append(f'id: {event_id}')
    if event:
        lines.append(f'event: {event}')
    lines.append(f'data: {json.dumps(data, ensure_ascii=False, default=str)}')
    return '\n'.join(lines) + '\n\n'


def stream_state():
    return {
        'output_parts': [],
        'tool_names': {},
        'tool_inputs': {},
        'started_tools': set(),
        'hidden_tools': set(),
    }


def hermes_sse(data=None, comment=None, event_id=None):
    if comment is not None:
        return f': {comment}\n\n'
    payload = json.dumps(data, ensure_ascii=False, default=str)
    if event_id:
        # 输出 SSE `id:` 行,客户端用 EventSource 时 lastEventId 自动回填,断线
        # 重连无需额外读 GET /v1/runs/{id} 拿 cursor。cursor schema 与 Redis
        # Stream 一致:`<13 位毫秒>-<序号>`,thread 模式合成同样格式以便客户端
        # 不区分后端实现。
        return f'id: {event_id}\ndata: {payload}\n\n'
    return f'data: {payload}\n\n'


_THREAD_CURSOR_LOCK = threading.Lock()
_THREAD_CURSOR_LAST = {'ms': 0, 'seq': 0}


def _next_thread_cursor():
    """生成 thread 模式下兼容 Redis Stream 形态的 event id (`<ms>-<seq>`)。

    并发安全;单进程内单调递增。同毫秒内序号自增,跨毫秒重置 seq。"""
    with _THREAD_CURSOR_LOCK:
        now_ms = int(time.time() * 1000)
        if now_ms <= _THREAD_CURSOR_LAST['ms']:
            now_ms = _THREAD_CURSOR_LAST['ms']
            _THREAD_CURSOR_LAST['seq'] += 1
        else:
            _THREAD_CURSOR_LAST['ms'] = now_ms
            _THREAD_CURSOR_LAST['seq'] = 0
        return f"{now_ms}-{_THREAD_CURSOR_LAST['seq']}"


def run_completed_payload(run_id, session_id, output, usage=None):
    return {
        'event': 'run.completed',
        'run_id': run_id,
        'timestamp': time.time(),
        'output': output or '',
        'usage': usage,
    }


def run_failed_payload(run_id, error):
    return {
        'event': 'run.failed',
        'run_id': run_id,
        'timestamp': time.time(),
        'error': error,
    }


def text_from_update(update):
    content = update.get('content') or {}
    return (content.get('text') or '') if isinstance(content, dict) else ''


def step_id_from_tool_id(tool_id):
    match = re.match(r'call_(\d+)_', tool_id or '')
    return f'model-{match.group(1)}' if match else ''


def next_generated_step_id(state, prefix='reasoning'):
    state['generated_step_counter'] = state.get('generated_step_counter', 0) + 1
    return f'{prefix}-{state["generated_step_counter"]}'


def hermes_events_from_update(run_id, update, state):
    event_type = update.get('sessionUpdate')
    timestamp = time.time()

    if event_type == 'agent_message_chunk':
        text = text_from_update(update)
        if text:
            state['output_parts'].append(text)
            return [{'event': 'message.delta', 'run_id': run_id, 'timestamp': timestamp, 'delta': text}]
        return []

    if event_type == 'thought_start':
        step_id = update.get('thoughtId') or next_generated_step_id(state)
        state['current_step_id'] = step_id
        evt = {
            'event': 'reasoning.started',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'title': update.get('title') or 'Agent step',
            'status': update.get('status') or 'in_progress',
        }
        # hidden/replace 等"提示位"字段只在 truthy 时下发,默认 false 是噪音——
        # 客户端若没看到字段就当未设置即可。
        if update.get('hidden'):
            evt['hidden'] = True
        return [evt]

    if event_type == 'thought_delta':
        step_id = update.get('thoughtId') or state.get('current_step_id') or next_generated_step_id(state)
        text = text_from_update(update)
        if text:
            evt = {
                'event': 'reasoning.available',
                'run_id': run_id,
                'timestamp': timestamp,
                'step_id': step_id,
                'text': text,
            }
            if update.get('replace'):
                evt['replace'] = True
            return [evt]
        return []

    if event_type == 'thought_done':
        step_id = update.get('thoughtId') or state.get('current_step_id') or next_generated_step_id(state)
        if state.get('current_step_id') == step_id:
            state['current_step_id'] = ''
        evt = {
            'event': 'reasoning.completed',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'status': update.get('status') or 'completed',
            'text': text_from_update(update),
        }
        if update.get('hidden'):
            evt['hidden'] = True
        return [evt]

    if event_type == 'thought':
        step_id = next_generated_step_id(state, prefix='note')
        text = text_from_update(update)
        events = [{
            'event': 'reasoning.started',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'title': update.get('title') or 'Thinking',
            'status': 'in_progress',
        }]
        if text:
            events.append({
                'event': 'reasoning.available',
                'run_id': run_id,
                'timestamp': timestamp,
                'step_id': step_id,
                'text': text,
            })
        events.append({
            'event': 'reasoning.completed',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'status': 'completed',
        })
        return events

    if event_type == 'tool_call_delta':
        tool_id = update.get('toolCallId') or f"call_{update.get('index', len(state['tool_names']))}"
        tool_name = update.get('name') or state['tool_names'].get(tool_id) or 'tool'
        if update.get('hidden'):
            state['hidden_tools'].add(tool_id)
        state['tool_names'][tool_id] = tool_name
        if update.get('argumentsText'):
            state.setdefault('tool_inputs', {})[tool_id] = update.get('argumentsText')
        evt = {
            'event': 'tool.delta',
            'run_id': run_id,
            'timestamp': timestamp,
            'tool_call_id': tool_id,
            'step_id': step_id_from_tool_id(tool_id) or state.get('current_step_id') or '',
            'tool': tool_name,
            'preview': update.get('title') or tool_name,
            'status': update.get('status') or 'in_progress',
            'arguments_delta': update.get('argumentsDelta') or '',
            'arguments_text': update.get('argumentsText') or '',
        }
        # 上游若没标 kind 就别填一个 'tool' 假值——避免下游把它当成"未知工具"误判;
        # hidden 同理,默认 false 是噪音,只在 truthy 时下发。
        if update.get('kind'):
            evt['kind'] = update.get('kind')
        if update.get('hidden'):
            evt['hidden'] = True
        return [evt]

    if event_type == 'tool_call':
        tool_id = update.get('toolCallId') or f"call_{len(state['tool_names'])}"
        tool_name = update.get('name') or state['tool_names'].get(tool_id) or 'tool'
        if update.get('hidden'):
            state['hidden_tools'].add(tool_id)
        state['tool_names'][tool_id] = tool_name
        if update.get('input') is not None:
            state.setdefault('tool_inputs', {})[tool_id] = update.get('input')
        if tool_id in state['started_tools']:
            return []
        state['started_tools'].add(tool_id)
        preview = update.get('title') or tool_name
        evt = {
            'event': 'tool.started',
            'run_id': run_id,
            'timestamp': timestamp,
            'tool_call_id': tool_id,
            'step_id': step_id_from_tool_id(tool_id) or state.get('current_step_id') or '',
            'tool': tool_name,
            'preview': preview,
            'status': update.get('status') or 'in_progress',
            'input': state.get('tool_inputs', {}).get(tool_id, update.get('input')),
        }
        if update.get('kind'):
            evt['kind'] = update.get('kind')
        if update.get('hidden'):
            evt['hidden'] = True
        return [evt]

    if event_type == 'tool_call_update':
        status = update.get('status')
        tool_id = update.get('toolCallId') or ''
        if tool_id in state['hidden_tools']:
            return []
        tool_name = state['tool_names'].get(tool_id) or 'tool'
        if update.get('input') is not None:
            state.setdefault('tool_inputs', {})[tool_id] = update.get('input')
        if status not in ('completed', 'failed'):
            return [{
                'event': 'tool.updated',
                'run_id': run_id,
                'timestamp': timestamp,
                'tool_call_id': tool_id,
                'step_id': step_id_from_tool_id(tool_id) or '',
                'tool': tool_name,
                'status': status or 'in_progress',
                'content': text_from_update(update),
                'data': update.get('data'),
            }]
        return [{
            'event': 'tool.completed',
            'run_id': run_id,
            'timestamp': timestamp,
            'tool_call_id': tool_id,
            'step_id': step_id_from_tool_id(tool_id) or '',
            'tool': tool_name,
            'duration': 0,
            'status': status,
            'error': status == 'failed',
            'content': text_from_update(update),
            'data': update.get('data'),
        }]

    if event_type == 'ask_user':
        return [{
            'event': 'ask_user',
            'run_id': run_id,
            'timestamp': timestamp,
            'question': update.get('question') or '',
            'candidates': update.get('candidates') or [],
        }]

    return []


def run_response_payload(session_id, run_id, status='started', cursor='0-0', regenerated_from_run_id=''):
    return {
        'id': run_id,
        'object': 'agent.run',
        'run_id': run_id,
        'session_id': session_id,
        'status': status,
        'cursor': cursor,
        **({'regenerated_from_run_id': regenerated_from_run_id} if regenerated_from_run_id else {}),
    }


def response_id():
    return f'resp_{uuid.uuid4().hex}'


def response_object(resp_id, model, status='completed', output=None, created_at=None, error=None, usage=None):
    payload = {
        'id': resp_id,
        'object': 'response',
        'created_at': created_at or int(time.time()),
        'status': status,
        'model': model,
        'output': output or [],
        'usage': dict(usage) if usage else None,
    }
    if error:
        payload['error'] = error
    return payload


def response_text_item(text):
    return {
        'id': f'msg_{uuid.uuid4().hex}',
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': text or ''}],
    }


def response_function_call_item(event):
    raw_arguments = event.get('input')
    if isinstance(raw_arguments, str):
        arguments = raw_arguments
    else:
        arguments = json.dumps(raw_arguments or {}, ensure_ascii=False, default=str)
    return {
        'id': f'fc_{uuid.uuid4().hex}',
        'type': 'function_call',
        'status': 'completed',
        'call_id': event.get('tool_call_id') or '',
        'name': event.get('tool') or 'tool',
        'arguments': arguments,
    }


def response_function_output_item(event):
    output = event.get('content') or event.get('data')
    if not isinstance(output, str):
        output = json.dumps(output or {}, ensure_ascii=False, default=str)
    return {
        'id': f'fco_{uuid.uuid4().hex}',
        'type': 'function_call_output',
        'status': 'completed',
        'call_id': event.get('tool_call_id') or '',
        'output': output,
    }


def response_sse(event_type, data, sequence_number):
    payload = dict(data or {})
    payload.setdefault('type', event_type)
    payload.setdefault('sequence_number', sequence_number)
    return sse_encode(payload, event=event_type)


def content_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get('text'), str):
            return value.get('text') or ''
        if isinstance(value.get('content'), (str, list, dict)):
            return content_text(value.get('content'))
        return ''
    if isinstance(value, list):
        if any(isinstance(item, dict) and 'role' in item for item in value):
            text = last_user_text(value)
            if text:
                return text
            return content_text(value[-1].get('content') if value and isinstance(value[-1], dict) else '')
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get('type') in ('text', 'input_text', 'output_text') and isinstance(item.get('text'), str):
                parts.append(item.get('text') or '')
            else:
                parts.append(content_text(item))
        return '\n'.join(part for part in parts if part)
    return str(value)


def prompt_text_from_body(body):
    for key in ('message', 'text', 'prompt', 'input'):
        text = content_text(body.get(key))
        if text.strip():
            return text
    return content_text(body.get('messages') or [])


def start_celery_run(session_id, user_id, text, mode='events', cwd=None, model=None):
    if celery_service is None:
        raise HTTPException(status_code=500, detail='Celery runner is not enabled')
    try:
        result = celery_service.start_or_answer(session_id, user_id, text, mode=mode, cwd=cwd, model=model)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


def start_thread_run(session_id, user_id, text, cwd=None, model=None):
    try:
        sess = service.get_session(session_id, user_id=user_id, cwd=cwd)
        if sess is None:
            return None
        run_id = sess.run_or_answer(text, mode='events', model_override=model)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    with THREAD_RUNS_LOCK:
        THREAD_RUNS[run_id] = {
            'session_id': sess.sid,
            'user_id': user_id,
            'created_at': int(time.time()),
        }
    return {'session_id': sess.sid, 'run_id': run_id, 'cursor': '0-0'}


def start_agent_run(session_id, user_id, text, cwd=None, model=None):
    if celery_service is not None:
        run = start_celery_run(session_id, user_id, text, mode='events', cwd=cwd, model=model)
        return {'session_id': run.session_id, 'run_id': run.run_id, 'cursor': run.cursor}
    result = start_thread_run(session_id, user_id, text, cwd=cwd, model=model)
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


def load_run_record(run_id, user_id):
    if celery_service is not None:
        return celery_service.load_run(run_id, user_id)
    with THREAD_RUNS_LOCK:
        thread_run = THREAD_RUNS.get(run_id)
    if not thread_run or thread_run.get('user_id') != user_id:
        return None
    run = service.store.load_run(run_id, user_id=user_id) or {}
    return {
        'run_id': run_id,
        'session_id': thread_run.get('session_id') or run.get('session_id') or '',
        'status': run.get('status') or 'running',
        'error': run.get('error') or '',
        'created_at': run.get('created_at') or thread_run.get('created_at') or '',
        'updated_at': run.get('updated_at') or '',
        'started_at': run.get('started_at') or '',
        'finished_at': run.get('finished_at') or '',
        'last_event_id': run.get('last_event_id') or '',
        'mode': run.get('mode') or 'events',
    }


def ensure_session(session_id=None, cwd=None):
    if session_id:
        sess = service.get_session(session_id, user_id=SERVER_USER_ID, cwd=cwd)
    else:
        sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
    if sess is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return sess


def response_messages_from_input(raw_input):
    if raw_input is None:
        return []
    if isinstance(raw_input, str):
        return [{'role': 'user', 'content': raw_input}]
    if isinstance(raw_input, dict):
        role = raw_input.get('role') or 'user'
        return [{'role': role, 'content': content_text(raw_input.get('content', raw_input))}]
    if isinstance(raw_input, list):
        messages = []
        for item in raw_input:
            if isinstance(item, str):
                messages.append({'role': 'user', 'content': item})
            elif isinstance(item, dict) and 'role' in item:
                messages.append({'role': item.get('role') or 'user', 'content': content_text(item.get('content'))})
            else:
                text = content_text(item)
                if text:
                    messages.append({'role': 'user', 'content': text})
        return messages
    return [{'role': 'user', 'content': str(raw_input)}]


def response_prompt_from_messages(messages, instructions='', include_history=False):
    user_text = next(
        ((message.get('content') or '').strip() for message in reversed(messages) if message.get('role') == 'user'),
        '',
    )
    if not user_text:
        user_text = '\n'.join((message.get('content') or '').strip() for message in messages if message.get('content')).strip()
    if not include_history:
        return user_text
    parts = []
    if instructions:
        parts.append(f'Instructions:\n{instructions.strip()}')
    history = [message for message in messages[:-1] if message.get('content')]
    if history:
        parts.append('Conversation history:\n' + json.dumps(history, ensure_ascii=False, default=str))
    parts.append(user_text)
    return '\n\n'.join(part for part in parts if part)


def build_response_from_run_events(resp_id, model, created_at, run_id, updates):
    state = stream_state()
    output = []
    final_text = ''
    usage = None
    for update in updates:
        for event in hermes_events_from_update(run_id, update, state):
            if event.get('event') == 'tool.started':
                output.append(response_function_call_item(event))
            elif event.get('event') == 'tool.completed':
                output.append(response_function_output_item(event))
        if update.get('sessionUpdate') == 'done':
            usage = update.get('usage') or None
            break
    final_text = ''.join(state['output_parts'])
    if final_text:
        output.append(response_text_item(final_text))
    return response_object(resp_id, model, output=output, created_at=created_at, usage=usage), final_text


def iter_run_updates(run):
    if celery_service is not None:
        yield from celery_service.iter_events(run['run_id'], run.get('cursor', '0-0'))
        return
    sess = service.load_session(run['session_id'], user_id=SERVER_USER_ID)
    if sess is None:
        raise HTTPException(status_code=404, detail='Session not found')
    while True:
        try:
            item = sess.display_q.get(timeout=1)
        except queue.Empty:
            if sess.turn_done_evt.is_set():
                return
            continue
        if 'event' not in item:
            if sess.turn_done_evt.is_set() and sess.display_q.empty():
                return
            continue
        update = item['event']
        yield update
        if update.get('sessionUpdate') == 'done':
            return


def save_response_record(resp_id, response, conversation_history, instructions='', session_id='', conversation='', store=True):
    if not store:
        return
    service.store.save_response(
        resp_id,
        response,
        conversation_history=conversation_history,
        instructions=instructions,
        session_id=session_id,
        conversation=conversation,
        user_id=SERVER_USER_ID,
    )


def start_celery_regenerate(session_id, user_id, mode='events', cwd=None):
    if celery_service is None:
        raise HTTPException(status_code=500, detail='Celery runner is not enabled')
    try:
        result = celery_service.regenerate_last_answer(session_id, user_id, mode=mode, cwd=cwd)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


def start_thread_regenerate(session_id, user_id):
    try:
        result = service.regenerate_session(session_id, user_id=user_id)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    with THREAD_RUNS_LOCK:
        THREAD_RUNS[result['run_id']] = {
            'session_id': result['session_id'],
            'user_id': user_id,
            'created_at': int(time.time()),
        }
    return result


def start_agent_regenerate(session_id, user_id, cwd=None):
    if celery_service is not None:
        run = start_celery_regenerate(session_id, user_id, mode='events', cwd=cwd)
        return {
            'session_id': run.session_id,
            'run_id': run.run_id,
            'cursor': run.cursor,
            'regenerated_from_run_id': run.regenerated_from_run_id,
        }
    return start_thread_regenerate(session_id, user_id)


async def celery_run_event_stream(request, run_id, session_id, user_id, last_id='0-0'):
    idle_started = time.monotonic()
    last_heartbeat = time.monotonic()
    state = stream_state()
    while True:
        if await request.is_disconnected():
            celery_service.cancel_run(run_id, user_id)
            break

        items = await anyio.to_thread.run_sync(
            lambda: celery_service.read_events(run_id, user_id, last_id=last_id, block_ms=1000)
        )
        if items is None:
            yield hermes_sse(run_failed_payload(run_id, 'Run not found'))
            yield hermes_sse(comment='stream closed')
            break
        if not items:
            now = time.monotonic()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield hermes_sse(comment='keepalive')
                last_heartbeat = now
            if now - idle_started > int(getattr(config, 'RUN_IDLE_TIMEOUT_SECONDS', 60 * 60)):
                yield hermes_sse(run_failed_payload(run_id, 'Run timed out'))
                yield hermes_sse(comment='stream closed')
                break
            continue

        idle_started = time.monotonic()
        last_heartbeat = idle_started
        for event_id, update in items:
            last_id = event_id
            hermes_list = list(hermes_events_from_update(run_id, update, state))
            for idx, event in enumerate(hermes_list):
                # 一个 redis-stream 条目可能展开成多条 hermes 事件,只把 id
                # 挂到最后一条上,这样客户端 lastEventId 始终对齐 redis cursor。
                pass_id = event_id if idx == len(hermes_list) - 1 else None
                yield hermes_sse(event, event_id=pass_id)
            if update.get('sessionUpdate') == 'done':
                yield hermes_sse(
                    run_completed_payload(
                        run_id, session_id, ''.join(state['output_parts']), usage=update.get('usage'),
                    ),
                    event_id=event_id,
                )
                yield hermes_sse(comment='stream closed')
                return


async def thread_run_event_stream(request, run_id, session_id, user_id):
    with THREAD_RUNS_LOCK:
        run = THREAD_RUNS.get(run_id)
    if not run or run.get('user_id') != user_id:
        raise HTTPException(status_code=404, detail='Run not found')
    sess = service.load_session(session_id, user_id=user_id)
    if sess is None:
        raise HTTPException(status_code=404, detail='Session not found')
    last_heartbeat = time.monotonic()
    state = stream_state()
    while True:
        if await request.is_disconnected():
            # 等待用户输入时客户端正常会断开，下一轮再连——这种情况不要 cancel session
            if sess.status != SESSION_WAITING_USER:
                service.cancel_session(session_id, user_id=user_id)
            break
        try:
            item = await anyio.to_thread.run_sync(lambda: sess.display_q.get(timeout=1))
        except queue.Empty:
            if sess.turn_done_evt.is_set():
                break
            now = time.monotonic()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield hermes_sse(comment='keepalive')
                last_heartbeat = now
            continue
        if 'event' not in item:
            if sess.turn_done_evt.is_set() and sess.display_q.empty():
                break
            continue
        update = item['event']
        hermes_list = list(hermes_events_from_update(run_id, update, state))
        for idx, event in enumerate(hermes_list):
            pass_id = _next_thread_cursor() if idx == len(hermes_list) - 1 else None
            yield hermes_sse(event, event_id=pass_id)
        last_heartbeat = time.monotonic()
        if update.get('sessionUpdate') == 'done':
            yield hermes_sse(
                run_completed_payload(
                    run_id, session_id, ''.join(state['output_parts']), usage=update.get('usage'),
                ),
                event_id=_next_thread_cursor(),
            )
            yield hermes_sse(comment='stream closed')
            break
        if sess.turn_done_evt.is_set() and sess.display_q.empty():
            break


async def openai_celery_stream(request, model, run_id, session_id, user_id, last_id='0-0'):
    yield sse_encode(openai_role_chunk(model))
    idle_started = time.monotonic()
    last_heartbeat = time.monotonic()
    state = stream_state()
    while True:
        if await request.is_disconnected():
            celery_service.cancel_run(run_id, user_id)
            break
        items = await anyio.to_thread.run_sync(
            lambda: celery_service.read_events(run_id, user_id, last_id=last_id, block_ms=1000)
        )
        if items is None:
            yield sse_encode(openai_done_chunk(model))
            yield 'data: [DONE]\n\n'
            break
        if not items:
            now = time.monotonic()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield sse_encode(comment='keepalive')
                last_heartbeat = now
            if now - idle_started > int(getattr(config, 'RUN_IDLE_TIMEOUT_SECONDS', 60 * 60)):
                yield sse_encode(openai_done_chunk(model))
                yield 'data: [DONE]\n\n'
                break
            continue
        idle_started = time.monotonic()
        last_heartbeat = idle_started
        for event_id, update in items:
            last_id = event_id
            for event in hermes_events_from_update(run_id, update, state):
                if event.get('event') == 'message.delta':
                    yield sse_encode(openai_chat_chunk(model, event.get('delta') or ''))
            if update.get('sessionUpdate') == 'done':
                yield sse_encode(openai_done_chunk(model, usage=update.get('usage')))
                yield 'data: [DONE]\n\n'
                return


async def openai_thread_stream(request, model, run_id, session_id, user_id):
    yield sse_encode(openai_role_chunk(model))
    with THREAD_RUNS_LOCK:
        run = THREAD_RUNS.get(run_id)
    if not run or run.get('user_id') != user_id:
        yield sse_encode(openai_done_chunk(model))
        yield 'data: [DONE]\n\n'
        return
    sess = service.load_session(session_id, user_id=user_id)
    if sess is None:
        yield sse_encode(openai_done_chunk(model))
        yield 'data: [DONE]\n\n'
        return
    last_heartbeat = time.monotonic()
    state = stream_state()
    while True:
        if await request.is_disconnected():
            if sess.status != SESSION_WAITING_USER:
                service.cancel_session(session_id, user_id=user_id)
            break
        try:
            item = await anyio.to_thread.run_sync(lambda: sess.display_q.get(timeout=1))
        except queue.Empty:
            if sess.turn_done_evt.is_set():
                break
            now = time.monotonic()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield sse_encode(comment='keepalive')
                last_heartbeat = now
            continue
        if 'event' not in item:
            if sess.turn_done_evt.is_set() and sess.display_q.empty():
                break
            continue
        update = item['event']
        for event in hermes_events_from_update(run_id, update, state):
            if event.get('event') == 'message.delta':
                yield sse_encode(openai_chat_chunk(model, event.get('delta') or ''))
        last_heartbeat = time.monotonic()
        if update.get('sessionUpdate') == 'done':
            yield sse_encode(openai_done_chunk(model, usage=update.get('usage')))
            yield 'data: [DONE]\n\n'
            return
    yield sse_encode(openai_done_chunk(model))
    yield 'data: [DONE]\n\n'


async def response_stream(request, run, resp_id, model, created_at, conversation_history, instructions, conversation, store):
    sequence = 0
    state = stream_state()
    output = []
    final_text = ''
    message_started = False
    completed = False

    def emit(event_type, data):
        nonlocal sequence
        sequence += 1
        return response_sse(event_type, data, sequence)

    yield emit('response.created', response_object(resp_id, model, status='in_progress', created_at=created_at))

    async def handle_update(update):
        nonlocal message_started, final_text, completed
        chunks = []
        for event in hermes_events_from_update(run['run_id'], update, state):
            if event.get('event') == 'message.delta':
                if not message_started:
                    message_started = True
                    chunks.append(emit('response.output_item.added', {
                        'response_id': resp_id,
                        'output_index': len(output),
                        'item': {
                            'id': f'msg_{uuid.uuid4().hex}',
                            'type': 'message',
                            'status': 'in_progress',
                            'role': 'assistant',
                            'content': [],
                        },
                    }))
                chunks.append(emit('response.output_text.delta', {
                    'response_id': resp_id,
                    'delta': event.get('delta') or '',
                    'output_index': len(output),
                    'content_index': 0,
                }))
            elif event.get('event') == 'tool.started':
                item = response_function_call_item(event)
                output.append(item)
                chunks.append(emit('response.output_item.added', {
                    'response_id': resp_id,
                    'output_index': len(output) - 1,
                    'item': item,
                }))
                chunks.append(emit('response.output_item.done', {
                    'response_id': resp_id,
                    'output_index': len(output) - 1,
                    'item': item,
                }))
            elif event.get('event') == 'tool.completed':
                item = response_function_output_item(event)
                output.append(item)
                chunks.append(emit('response.output_item.added', {
                    'response_id': resp_id,
                    'output_index': len(output) - 1,
                    'item': item,
                }))
                chunks.append(emit('response.output_item.done', {
                    'response_id': resp_id,
                    'output_index': len(output) - 1,
                    'item': item,
                }))
        if update.get('sessionUpdate') == 'done':
            final_text = ''.join(state['output_parts'])
            if final_text:
                item = response_text_item(final_text)
                output.append(item)
                if message_started:
                    chunks.append(emit('response.output_text.done', {
                        'response_id': resp_id,
                        'text': final_text,
                        'output_index': len(output) - 1,
                        'content_index': 0,
                    }))
                    chunks.append(emit('response.output_item.done', {
                        'response_id': resp_id,
                        'output_index': len(output) - 1,
                        'item': item,
                    }))
            response = response_object(
                resp_id,
                model,
                output=output,
                created_at=created_at,
                usage=update.get('usage'),
            )
            history = list(conversation_history or [])
            if final_text:
                history.append({'role': 'assistant', 'content': final_text})
            save_response_record(
                resp_id,
                response,
                history,
                instructions=instructions,
                session_id=run['session_id'],
                conversation=conversation,
                store=store,
            )
            chunks.append(emit('response.completed', response))
            chunks.append('data: [DONE]\n\n')
            completed = True
        return chunks

    if celery_service is not None:
        last_id = run.get('cursor', '0-0')
        idle_started = time.monotonic()
        last_heartbeat = time.monotonic()
        while not completed:
            if await request.is_disconnected():
                celery_service.cancel_run(run['run_id'], SERVER_USER_ID)
                break
            items = await anyio.to_thread.run_sync(
                lambda: celery_service.read_events(run['run_id'], SERVER_USER_ID, last_id=last_id, block_ms=1000)
            )
            if items is None:
                yield emit('response.failed', response_object(
                    resp_id,
                    model,
                    status='failed',
                    output=output,
                    created_at=created_at,
                    error={'message': 'Run not found'},
                ))
                yield 'data: [DONE]\n\n'
                break
            if not items:
                now = time.monotonic()
                if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                    yield sse_encode(comment='keepalive')
                    last_heartbeat = now
                if now - idle_started > int(getattr(config, 'RUN_IDLE_TIMEOUT_SECONDS', 60 * 60)):
                    yield emit('response.failed', response_object(
                        resp_id,
                        model,
                        status='failed',
                        output=output,
                        created_at=created_at,
                        error={'message': 'Run timed out'},
                    ))
                    yield 'data: [DONE]\n\n'
                    break
                continue
            idle_started = time.monotonic()
            last_heartbeat = idle_started
            for event_id, update in items:
                last_id = event_id
                for chunk in await handle_update(update):
                    yield chunk
                if completed:
                    return
        return

    sess = service.load_session(run['session_id'], user_id=SERVER_USER_ID)
    if sess is None:
        yield emit('response.failed', response_object(
            resp_id,
            model,
            status='failed',
            output=output,
            created_at=created_at,
            error={'message': 'Session not found'},
        ))
        yield 'data: [DONE]\n\n'
        return
    last_heartbeat = time.monotonic()
    while not completed:
        if await request.is_disconnected():
            if sess.status != SESSION_WAITING_USER:
                service.cancel_session(run['session_id'], user_id=SERVER_USER_ID)
            break
        try:
            item = await anyio.to_thread.run_sync(lambda: sess.display_q.get(timeout=1))
        except queue.Empty:
            if sess.turn_done_evt.is_set():
                break
            now = time.monotonic()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield sse_encode(comment='keepalive')
                last_heartbeat = now
            continue
        if 'event' not in item:
            if sess.turn_done_evt.is_set() and sess.display_q.empty():
                break
            continue
        for chunk in await handle_update(item['event']):
            yield chunk
        last_heartbeat = time.monotonic()
        if completed:
            return


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/health/detailed')
def health_detailed():
    checks = {'api': 'ok', 'sqlite': 'unknown', 'redis': 'disabled', 'runner': RUNNER_BACKEND}
    try:
        with service.store._connect() as conn:
            conn.execute('SELECT 1').fetchone()
        checks['sqlite'] = 'ok'
    except Exception as e:
        checks['sqlite'] = f'error: {e}'
    if celery_service is not None:
        try:
            checks['redis'] = 'ok' if celery_service.bus.ping() else 'error'
        except Exception as e:
            checks['redis'] = f'error: {e}'
    status = 'ok' if all(value in ('ok', 'disabled', RUNNER_BACKEND) for value in checks.values()) else 'degraded'
    return {
        'status': status,
        'runner_backend': RUNNER_BACKEND,
        'model': runtime_config.get_active_model(),
        'checks': checks,
    }


@app.get('/v1/models')
def models():
    active = runtime_config.get_active_model()
    configured = getattr(config, 'MODEL', 'qwen-plus')
    aliases = list(getattr(config, 'MODEL_ALIASES', ['mini-agent', 'agent']) or [])
    ids = []
    for model_id in [active, configured, *aliases]:
        if model_id and model_id not in ids:
            ids.append(model_id)
    return {
        'object': 'list',
        'active_model': active,
        'data': [
            {
                'id': model_id,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'pai-rag',
                'active': model_id == active,
            }
            for model_id in ids
        ],
    }


@app.post('/v1/models/active')
async def set_active_model_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    try:
        new_model = runtime_config.set_active_model(body.get('model'))
    except ValueError as exc:
        return JSONResponse(openai_error(str(exc)), status_code=400)
    return {'active_model': new_model}


@app.get('/v1/skills')
def skills():
    scope = handler_memory_scope(SERVER_USER_ID)
    return skills_inventory(ROOT, scope.root)


@app.post('/v1/runs')
async def create_run(
    request: Request,
    x_session_id: str | None = Header(default=None),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    text = prompt_text_from_body(body)
    cwd = body.get('cwd')
    if not text.strip():
        message = "Missing 'input' field" if 'input' not in body else 'No user message found in input'
        return JSONResponse(openai_error(message), status_code=400)

    model_override = body.get('model')
    if model_override is not None:
        if not isinstance(model_override, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(model_override)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)

    session_id = body.get('session_id') or x_session_id
    if not session_id:
        try:
            sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
        except WorkspaceViolation as e:
            raise backend_error(e) from e
        session_id = sess.sid

    stream = bool(body.get('stream'))
    run = start_agent_run(session_id, SERVER_USER_ID, text, cwd=cwd, model=model_override)
    headers = stream_headers(run['session_id'], run['run_id'])
    if stream:
        if celery_service is not None:
            event_stream = celery_run_event_stream(
                request,
                run['run_id'],
                run['session_id'],
                SERVER_USER_ID,
                last_id=run.get('cursor', '0-0'),
            )
        else:
            event_stream = thread_run_event_stream(
                request,
                run['run_id'],
                run['session_id'],
                SERVER_USER_ID,
            )
        return StreamingResponse(event_stream, media_type='text/event-stream', headers=headers)
    payload = run_response_payload(run['session_id'], run['run_id'], cursor=run.get('cursor', '0-0'))
    return JSONResponse(payload, status_code=202, headers=headers)


@app.get('/v1/runs/{run_id}/events')
async def run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = None,
    last_event_id_header: str | None = Header(default=None, alias='Last-Event-ID'),
):
    if celery_service is not None:
        run = celery_service.load_run(run_id, SERVER_USER_ID)
        if run is None:
            return JSONResponse(openai_error(f'Run not found: {run_id}', code='run_not_found'), status_code=404)
        cursor = last_event_id or last_event_id_header or run.get('last_event_id') or '0-0'
        event_stream = celery_run_event_stream(
            request,
            run_id,
            run['session_id'],
            SERVER_USER_ID,
            last_id=cursor,
        )
        return StreamingResponse(
            event_stream,
            media_type='text/event-stream',
            headers=stream_headers(run['session_id'], run_id),
        )

    with THREAD_RUNS_LOCK:
        run = THREAD_RUNS.get(run_id)
    if not run or run.get('user_id') != SERVER_USER_ID:
        return JSONResponse(openai_error(f'Run not found: {run_id}', code='run_not_found'), status_code=404)
    return StreamingResponse(
        thread_run_event_stream(request, run_id, run['session_id'], SERVER_USER_ID),
        media_type='text/event-stream',
        headers=stream_headers(run['session_id'], run_id),
    )


@app.get('/v1/runs/{run_id}')
def get_run(run_id: str):
    run = load_run_record(run_id, SERVER_USER_ID)
    if run is None:
        return JSONResponse(openai_error(f'Run not found: {run_id}', code='run_not_found'), status_code=404)
    return {
        'id': run_id,
        'object': 'agent.run',
        'run_id': run_id,
        'session_id': run.get('session_id') or '',
        'status': run.get('status') or '',
        'mode': run.get('mode') or '',
        'error': run.get('error') or '',
        'created_at': run.get('created_at') or '',
        'updated_at': run.get('updated_at') or '',
        'started_at': run.get('started_at') or '',
        'finished_at': run.get('finished_at') or '',
        'last_event_id': run.get('last_event_id') or '',
    }


@app.post('/v1/runs/{run_id}/stop')
def stop_run(run_id: str):
    if celery_service is not None:
        stopped = celery_service.cancel_run(run_id, SERVER_USER_ID)
        if not stopped:
            return JSONResponse(openai_error(f'Run not found: {run_id}', code='run_not_found'), status_code=404)
        return {'run_id': run_id, 'status': 'stopping'}

    with THREAD_RUNS_LOCK:
        run = THREAD_RUNS.get(run_id)
    if not run or run.get('user_id') != SERVER_USER_ID:
        return JSONResponse(openai_error(f'Run not found: {run_id}', code='run_not_found'), status_code=404)
    stopped = service.cancel_session(run['session_id'], user_id=SERVER_USER_ID)
    if not stopped:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'run_id': run_id, 'status': 'stopping'}


@app.post('/v1/responses')
async def create_response(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)

    raw_model = body.get('model')
    model_override = None
    if raw_model:
        if not isinstance(raw_model, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(raw_model)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)
    model = model_override or runtime_config.get_active_model()
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    instructions = body.get('instructions') or ''
    conversation = body.get('conversation') or ''
    store = bool(body.get('store', True))
    previous_response_id = body.get('previous_response_id') or ''

    previous = None
    if previous_response_id:
        previous = service.store.load_response(previous_response_id, user_id=SERVER_USER_ID)
        if previous is None:
            return JSONResponse(openai_error(f'Response not found: {previous_response_id}', code='response_not_found'), status_code=404)
    elif conversation:
        latest = service.store.latest_response_for_conversation(conversation, user_id=SERVER_USER_ID)
        previous = service.store.load_response(latest, user_id=SERVER_USER_ID) if latest else None

    input_messages = response_messages_from_input(body.get('input'))
    if not input_messages:
        input_messages = response_messages_from_input(body.get('messages'))
    if not input_messages:
        return JSONResponse(openai_error("Missing 'input' field"), status_code=400)

    base_history = []
    if isinstance(body.get('conversation_history'), list):
        base_history = response_messages_from_input(body.get('conversation_history'))
    elif previous:
        base_history = list(previous.get('conversation_history') or [])
        if not instructions:
            instructions = previous.get('instructions') or ''
        if not conversation:
            conversation = previous.get('conversation') or ''

    session_id = body.get('session_id') or (previous or {}).get('session_id') or ''
    try:
        sess = ensure_session(session_id or None, cwd=cwd)
    except WorkspaceViolation as e:
        raise backend_error(e) from e

    history_for_store = [*base_history, *input_messages]
    task_text = response_prompt_from_messages(
        history_for_store,
        instructions=instructions,
        include_history=bool(base_history and not session_id),
    )
    if not task_text.strip():
        return JSONResponse(openai_error('No user message found in input'), status_code=400)

    run = start_agent_run(sess.sid, SERVER_USER_ID, task_text, cwd=cwd, model=model_override)
    resp_id = response_id()
    created_at = int(time.time())
    headers = stream_headers(run['session_id'], run['run_id'])
    if stream:
        return StreamingResponse(
            response_stream(
                request,
                run,
                resp_id,
                model,
                created_at,
                history_for_store,
                instructions,
                conversation,
                store,
            ),
            media_type='text/event-stream',
            headers=headers,
        )

    updates = list(iter_run_updates(run))
    response, final_text = build_response_from_run_events(resp_id, model, created_at, run['run_id'], updates)
    history = list(history_for_store)
    if final_text:
        history.append({'role': 'assistant', 'content': final_text})
    save_response_record(
        resp_id,
        response,
        history,
        instructions=instructions,
        session_id=run['session_id'],
        conversation=conversation,
        store=store,
    )
    return JSONResponse(response, headers=headers)


@app.get('/v1/responses/{resp_id}')
def get_response(resp_id: str):
    record = service.store.load_response(resp_id, user_id=SERVER_USER_ID)
    if record is None:
        return JSONResponse(openai_error(f'Response not found: {resp_id}', code='response_not_found'), status_code=404)
    return record.get('response') or {}


@app.delete('/v1/responses/{resp_id}')
def delete_response(resp_id: str):
    deleted = service.store.delete_response(resp_id, user_id=SERVER_USER_ID)
    if not deleted:
        return JSONResponse(openai_error(f'Response not found: {resp_id}', code='response_not_found'), status_code=404)
    return {'id': resp_id, 'object': 'response.deleted', 'deleted': True}


@app.post('/v1/chat/completions')
async def chat_completions(
    request: Request,
    x_session_id: str | None = Header(default=None),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    raw_model = body.get('model')
    model_override = None
    if raw_model:
        if not isinstance(raw_model, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(raw_model)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)
    model = model_override or runtime_config.get_active_model()
    messages = body.get('messages') or []
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    if not last_user_text(messages).strip():
        raise HTTPException(status_code=400, detail='messages must include a non-empty user message')

    user_text = last_user_text(messages)
    if celery_service is not None:
        if x_session_id:
            session_id = x_session_id
        else:
            try:
                sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
            except WorkspaceViolation as e:
                raise backend_error(e) from e
            session_id = sess.sid
        if stream:
            run = start_celery_run(session_id, SERVER_USER_ID, user_text, mode='events', cwd=cwd, model=model_override)
            return StreamingResponse(
                openai_celery_stream(request, model, run.run_id, run.session_id, SERVER_USER_ID, last_id=run.cursor),
                media_type='text/event-stream',
                headers=stream_headers(run.session_id, run.run_id),
            )
        run = start_celery_run(session_id, SERVER_USER_ID, user_text, mode='text', cwd=cwd, model=model_override)
        headers = stream_headers(run.session_id, run.run_id)
        response_session_id = run.session_id
        captured_usage: dict = {}

        def text_events():
            for update in celery_service.iter_events(run.run_id, run.cursor):
                if update.get('sessionUpdate') == 'agent_message_chunk':
                    yield ((update.get('content') or {}).get('text') or '')
                elif update.get('sessionUpdate') == 'done':
                    captured_usage.update(update.get('usage') or {})

        text_iter = text_events()
        usage_source = lambda: captured_usage  # noqa: E731
    else:
        if stream:
            try:
                sess = ensure_session(x_session_id, cwd=cwd)
            except WorkspaceViolation as e:
                raise backend_error(e) from e
            run = start_agent_run(sess.sid, SERVER_USER_ID, user_text, cwd=cwd, model=model_override)
            return StreamingResponse(
                openai_thread_stream(request, model, run['run_id'], run['session_id'], SERVER_USER_ID),
                media_type='text/event-stream',
                headers=stream_headers(run['session_id'], run['run_id']),
            )
        try:
            sess, text_iter = service.chat_text(x_session_id, messages, user_id=SERVER_USER_ID, cwd=cwd, model_override=model_override)
        except (SessionBusyError, ServiceCapacityError, WorkspaceViolation, ToolWorkspaceViolation) as e:
            raise backend_error(e) from e
        if sess is None:
            raise HTTPException(status_code=404, detail='Session not found')
        headers = stream_headers(sess.sid, getattr(sess, 'active_run_id', '') or '')
        response_session_id = sess.sid
        usage_source = lambda: (getattr(sess, 'exit_reason', None) or {}).get('usage') or {}  # noqa: E731

    content = ''.join(text_iter)
    payload = openai_chat_completion(model, content, response_session_id, usage=usage_source())
    return JSONResponse(payload, headers=headers)


@app.get('/v1/sessions')
def list_sessions():
    return {'object': 'list', 'data': service.list_sessions(user_id=SERVER_USER_ID)}


@app.post('/v1/sessions')
async def create_session(request: Request):
    body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    try:
        sess = service.create_session(user_id=SERVER_USER_ID, cwd=(body or {}).get('cwd'))
    except WorkspaceViolation as e:
        raise backend_error(e) from e
    loaded = service.store.load(sess.sid, user_id=SERVER_USER_ID) or {}
    return {
        'session_id': sess.sid,
        'title': loaded.get('title', 'New Task'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'status': loaded.get('status', 'idle'),
        'active_run_id': loaded.get('active_run_id', ''),
        'messages': loaded.get('ui_messages', []),
    }


@app.get('/v1/sessions/{session_id}')
def get_session(session_id: str):
    loaded = service.store.load(session_id, user_id=SERVER_USER_ID)
    if loaded is None:
        raise HTTPException(status_code=404, detail='Session not found')
    payload = {
        'session_id': session_id,
        'title': loaded.get('title', 'New Task'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'status': loaded.get('status', 'idle'),
        'active_run_id': loaded.get('active_run_id', ''),
        'messages': loaded.get('ui_messages', []),
    }
    if getattr(config, 'EXPOSE_SESSION_DEBUG', False):
        payload['llm_history'] = loaded.get('llm_history', [])
        payload['handler_state'] = loaded.get('handler_state')
    return payload


@app.post('/v1/sessions/{session_id}/regenerate')
async def regenerate_session_answer(session_id: str, request: Request):
    try:
        body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    run = start_agent_regenerate(session_id, SERVER_USER_ID, cwd=body.get('cwd'))
    payload = run_response_payload(
        run['session_id'],
        run['run_id'],
        cursor=run.get('cursor', '0-0'),
        regenerated_from_run_id=run.get('regenerated_from_run_id') or '',
    )
    return JSONResponse(payload, status_code=202, headers=stream_headers(run['session_id'], run['run_id']))


@app.delete('/v1/sessions/{session_id}')
def delete_session(session_id: str):
    deleted = service.delete_session(session_id, user_id=SERVER_USER_ID)
    if not deleted:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'deleted': True, 'session_id': session_id}


@app.post('/v1/sessions/{session_id}/cancel')
def cancel_session(session_id: str):
    if celery_service is not None:
        cancelled = celery_service.cancel_session(session_id, user_id=SERVER_USER_ID)
    else:
        cancelled = service.cancel_session(session_id, user_id=SERVER_USER_ID)
    if not cancelled:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'cancelled': True, 'session_id': session_id}
