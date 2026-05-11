import json
import os
import queue
import re
import threading
import time

import anyio
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agent_service import (
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


def hermes_sse(data=None, comment=None):
    if comment is not None:
        return f': {comment}\n\n'
    return f'data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n'


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
    match = re.match(r'tool-(\d+)-', tool_id or '')
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
        return [{
            'event': 'reasoning.started',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'title': update.get('title') or 'Agent step',
            'status': update.get('status') or 'in_progress',
            'hidden': bool(update.get('hidden')),
        }]

    if event_type == 'thought_delta':
        step_id = update.get('thoughtId') or state.get('current_step_id') or next_generated_step_id(state)
        text = text_from_update(update)
        if text:
            return [{
                'event': 'reasoning.available',
                'run_id': run_id,
                'timestamp': timestamp,
                'step_id': step_id,
                'text': text,
                'replace': bool(update.get('replace')),
            }]
        return []

    if event_type == 'thought_done':
        step_id = update.get('thoughtId') or state.get('current_step_id') or next_generated_step_id(state)
        if state.get('current_step_id') == step_id:
            state['current_step_id'] = ''
        return [{
            'event': 'reasoning.completed',
            'run_id': run_id,
            'timestamp': timestamp,
            'step_id': step_id,
            'status': update.get('status') or 'completed',
            'hidden': bool(update.get('hidden')),
            'text': text_from_update(update),
        }]

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
        tool_id = update.get('toolCallId') or f"tool-{update.get('index', len(state['tool_names']))}"
        tool_name = update.get('name') or state['tool_names'].get(tool_id) or 'tool'
        if update.get('hidden'):
            state['hidden_tools'].add(tool_id)
        state['tool_names'][tool_id] = tool_name
        return [{
            'event': 'tool.delta',
            'run_id': run_id,
            'timestamp': timestamp,
            'tool_call_id': tool_id,
            'step_id': step_id_from_tool_id(tool_id) or state.get('current_step_id') or '',
            'tool': tool_name,
            'preview': update.get('title') or tool_name,
            'kind': update.get('kind') or 'tool',
            'status': update.get('status') or 'in_progress',
            'hidden': bool(update.get('hidden')),
            'arguments_delta': update.get('argumentsDelta') or '',
            'arguments_text': update.get('argumentsText') or '',
        }]

    if event_type == 'tool_call':
        tool_id = update.get('toolCallId') or f"tool-{len(state['tool_names'])}"
        tool_name = update.get('name') or state['tool_names'].get(tool_id) or 'tool'
        if update.get('hidden'):
            state['hidden_tools'].add(tool_id)
        state['tool_names'][tool_id] = tool_name
        if tool_id in state['started_tools']:
            return []
        state['started_tools'].add(tool_id)
        preview = update.get('title') or tool_name
        return [{
            'event': 'tool.started',
            'run_id': run_id,
            'timestamp': timestamp,
            'tool_call_id': tool_id,
            'step_id': step_id_from_tool_id(tool_id) or state.get('current_step_id') or '',
            'tool': tool_name,
            'preview': preview,
            'kind': update.get('kind') or 'tool',
            'status': update.get('status') or 'in_progress',
            'hidden': bool(update.get('hidden')),
            'input': update.get('input'),
        }]

    if event_type == 'tool_call_update':
        status = update.get('status')
        tool_id = update.get('toolCallId') or ''
        if tool_id in state['hidden_tools']:
            return []
        tool_name = state['tool_names'].get(tool_id) or 'tool'
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


def run_response_payload(session_id, run_id, status='started', stream_from='0-0', regenerated_from_run_id=''):
    return {
        'id': run_id,
        'object': 'agent.run',
        'run_id': run_id,
        'session_id': session_id,
        'status': status,
        'stream_from': stream_from,
        **({'regenerated_from_run_id': regenerated_from_run_id} if regenerated_from_run_id else {}),
    }


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
            if isinstance(item, dict) and item.get('type') in ('text', 'input_text') and isinstance(item.get('text'), str):
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


def start_celery_run(session_id, user_id, text, mode='events', cwd=None):
    if celery_service is None:
        raise HTTPException(status_code=500, detail='Celery runner is not enabled')
    try:
        result = celery_service.start_or_answer(session_id, user_id, text, mode=mode, cwd=cwd)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


def start_thread_run(session_id, user_id, text, cwd=None):
    try:
        sess = service.get_session(session_id, user_id=user_id, cwd=cwd)
        if sess is None:
            return None
        run_id = sess.run_or_answer(text, mode='events')
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    with THREAD_RUNS_LOCK:
        THREAD_RUNS[run_id] = {
            'session_id': sess.sid,
            'user_id': user_id,
            'created_at': int(time.time()),
        }
    return {'session_id': sess.sid, 'run_id': run_id, 'stream_from': '0-0'}


def start_agent_run(session_id, user_id, text, cwd=None):
    if celery_service is not None:
        run = start_celery_run(session_id, user_id, text, mode='events', cwd=cwd)
        return {'session_id': run.session_id, 'run_id': run.run_id, 'stream_from': run.stream_from}
    result = start_thread_run(session_id, user_id, text, cwd=cwd)
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


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
            'stream_from': run.stream_from,
            'regenerated_from_run_id': run.regenerated_from_run_id,
        }
    return start_thread_regenerate(session_id, user_id)


async def celery_run_event_stream(request, run_id, session_id, user_id, last_id='0-0'):
    idle_started = time.monotonic()
    last_heartbeat = time.monotonic()
    state = {'output_parts': [], 'tool_names': {}, 'started_tools': set(), 'hidden_tools': set()}
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
            for event in hermes_events_from_update(run_id, update, state):
                yield hermes_sse(event)
            if update.get('sessionUpdate') == 'done':
                yield hermes_sse(run_completed_payload(run_id, session_id, ''.join(state['output_parts'])))
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
    state = {'output_parts': [], 'tool_names': {}, 'started_tools': set(), 'hidden_tools': set()}
    while True:
        if await request.is_disconnected():
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
        for event in hermes_events_from_update(run_id, update, state):
            yield hermes_sse(event)
        last_heartbeat = time.monotonic()
        if update.get('sessionUpdate') == 'done':
            yield hermes_sse(run_completed_payload(run_id, session_id, ''.join(state['output_parts'])))
            yield hermes_sse(comment='stream closed')
            break
        if sess.turn_done_evt.is_set() and sess.display_q.empty():
            break


async def openai_celery_stream(request, model, run_id, session_id, user_id, last_id='0-0'):
    yield sse_encode(openai_role_chunk(model))
    idle_started = time.monotonic()
    last_heartbeat = time.monotonic()
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
            if update.get('sessionUpdate') == 'agent_message_chunk':
                text = ((update.get('content') or {}).get('text') or '')
                if text:
                    yield sse_encode(openai_chat_chunk(model, text))
            if update.get('sessionUpdate') == 'done':
                yield sse_encode(openai_done_chunk(model))
                yield 'data: [DONE]\n\n'
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
        'model': getattr(config, 'MODEL', 'qwen-plus'),
        'checks': checks,
    }


@app.get('/v1/models')
def models():
    configured = getattr(config, 'MODEL', 'qwen-plus')
    aliases = list(getattr(config, 'MODEL_ALIASES', ['mini-agent', 'agent']) or [])
    ids = []
    for model_id in [configured, *aliases]:
        if model_id and model_id not in ids:
            ids.append(model_id)
    return {
        'object': 'list',
        'data': [
            {'id': model_id, 'object': 'model', 'created': 0, 'owned_by': 'pai-rag'}
            for model_id in ids
        ],
    }


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

    session_id = body.get('session_id') or x_session_id
    if not session_id:
        try:
            sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
        except WorkspaceViolation as e:
            raise backend_error(e) from e
        session_id = sess.sid

    run = start_agent_run(session_id, SERVER_USER_ID, text, cwd=cwd)
    payload = run_response_payload(run['session_id'], run['run_id'], stream_from=run.get('stream_from', '0-0'))
    return JSONResponse(payload, status_code=202, headers=stream_headers(run['session_id'], run['run_id']))


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
        stream_from = last_event_id or last_event_id_header or run.get('last_event_id') or '0-0'
        event_stream = celery_run_event_stream(
            request,
            run_id,
            run['session_id'],
            SERVER_USER_ID,
            last_id=stream_from,
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


@app.post('/v1/chat/completions')
async def chat_completions(
    request: Request,
    x_session_id: str | None = Header(default=None),
):
    body = await request.json()
    model = body.get('model') or getattr(config, 'MODEL', 'qwen-plus')
    messages = body.get('messages') or []
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    if not last_user_text(messages).strip():
        raise HTTPException(status_code=400, detail='messages must include a non-empty user message')

    if celery_service is not None:
        if x_session_id:
            session_id = x_session_id
        else:
            try:
                sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
            except WorkspaceViolation as e:
                raise backend_error(e) from e
            session_id = sess.sid
        run = start_celery_run(session_id, SERVER_USER_ID, last_user_text(messages), mode='text', cwd=cwd)
        headers = stream_headers(run.session_id, run.run_id)
        response_session_id = run.session_id

        def text_events():
            for update in celery_service.iter_events(run.run_id, run.stream_from):
                if update.get('sessionUpdate') == 'agent_message_chunk':
                    yield ((update.get('content') or {}).get('text') or '')

        text_iter = text_events()
    else:
        try:
            sess, text_iter = service.chat_text(x_session_id, messages, user_id=SERVER_USER_ID, cwd=cwd)
        except (SessionBusyError, ServiceCapacityError, WorkspaceViolation, ToolWorkspaceViolation) as e:
            raise backend_error(e) from e
        if sess is None:
            raise HTTPException(status_code=404, detail='Session not found')
        headers = stream_headers(sess.sid, getattr(sess, 'active_run_id', '') or '')
        response_session_id = sess.sid

    if stream:
        if celery_service is not None:
            return StreamingResponse(
                openai_celery_stream(request, model, run.run_id, run.session_id, SERVER_USER_ID, last_id=run.stream_from),
                media_type='text/event-stream',
                headers=headers,
            )

        def event_stream():
            yield f'data: {json.dumps(openai_role_chunk(model), ensure_ascii=False)}\n\n'
            for text in text_iter:
                if not text:
                    continue
                chunk = openai_chat_chunk(model, text)
                yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
            yield f'data: {json.dumps(openai_done_chunk(model), ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(event_stream(), media_type='text/event-stream', headers=headers)

    content = ''.join(text_iter)
    payload = openai_chat_completion(model, content, response_session_id)
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
        stream_from=run.get('stream_from', '0-0'),
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
